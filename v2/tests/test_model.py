from pathlib import Path
import tempfile

import pytest
import torch
from torch.nn import functional as F

from pa_prover_v2.model import (
    ARCHITECTURE, LemmaDecoderLM, ModelConfig, RotaryEmbedding,
    load_checkpoint, save_checkpoint, sequence_log_probs,
)
from pa_prover_v2.tokenizer import BOS_ID, EOS_ID, MEM_ID, PAD_ID, ByteTokenizer


torch.set_num_threads(1)


def tiny_model(slots=2):
    torch.manual_seed(23)
    return LemmaDecoderLM(ModelConfig(
        d_model=24, n_heads=3, n_layers=2, lemma_layers=2,
        lemma_slots=slots, max_seq_len=64, ffn_multiplier=2, dropout=0,
    )).eval()


def test_utf8_fixed_vocabulary_and_fingerprint():
    tokenizer = ByteTokenizer()
    text = "∀x: x + 0 = x\n中文 🧮"
    ids = tokenizer.encode(text, bos=True, eos=True)
    assert ids[0] == BOS_ID and ids[-1] == EOS_ID
    assert tokenizer.decode([PAD_ID, MEM_ID, *ids]) == text
    assert tokenizer.encode("A") == [ord("A") + 4]
    assert len(tokenizer) == 260
    assert tokenizer.fingerprint() == ByteTokenizer().fingerprint()
    assert len(tokenizer.fingerprint()) == 64
    with pytest.raises(ValueError, match="outside"):
        tokenizer.decode([260])


def test_main_decoder_is_causal_and_has_no_encoder_or_cross_attention():
    model = tiny_model()
    first = torch.tensor([[1, 30, 31, 32, 33]])
    changed = torch.tensor([[1, 30, 31, 90, 91]])
    with torch.no_grad():
        before = model(first).logits
        after = model(changed).logits
    torch.testing.assert_close(before[:, :3], after[:, :3], rtol=0, atol=0)
    assert not torch.equal(before[:, 3:], after[:, 3:])
    assert not any(isinstance(module, (torch.nn.TransformerEncoder, torch.nn.TransformerDecoder))
                   for module in model.modules())
    assert not any("cross" in name or "thought" in name for name, _ in model.named_modules())


def test_rope_rotation_preserves_norm_and_relative_position():
    torch.manual_seed(1)
    rope = RotaryEmbedding(8)
    query, key = torch.randn(1, 2, 1, 8), torch.randn(1, 2, 1, 8)
    zero = rope(query, torch.tensor([[0]]))
    shifted = rope(query, torch.tensor([[7]]))
    torch.testing.assert_close(zero, query, rtol=0, atol=0)
    torch.testing.assert_close(shifted.square().sum(-1), query.square().sum(-1))
    first = (rope(query, torch.tensor([[2]])) * rope(key, torch.tensor([[5]]))).sum(-1)
    second = (rope(query, torch.tensor([[9]])) * rope(key, torch.tensor([[12]]))).sum(-1)
    torch.testing.assert_close(first, second)
    assert not torch.equal(shifted, query)


@pytest.mark.parametrize("slots", [1, 2, 4])
def test_one_or_more_summary_slots_preserve_length_and_receive_gradients(slots):
    model = tiny_model(slots)
    input_ids = torch.tensor([[BOS_ID, 40, *([MEM_ID] * slots), 41, EOS_ID]])
    memory = {(0, 2): [BOS_ID, 60, 61, EOS_ID]}
    labels = input_ids.clone()
    labels[:, :2 + slots] = -100
    output = model(input_ids, labels=labels, memories=memory)
    assert output.logits.shape == (*input_ids.shape, 260)
    assert output.loss.isfinite()
    output.loss.backward()
    assert model.summary_slots.grad.abs().sum() > 0
    assert model.lemma_embedding.weight.grad.abs().sum() > 0
    assert model.lemma_decoder.layers[0].attention.qkv.weight.grad.abs().sum() > 0
    assert model.decoder.layers[0].attention.qkv.weight.grad.abs().sum() > 0


def test_lemma_summary_slots_are_causal_and_depend_on_lemma_text():
    model = tiny_model(2)
    with torch.no_grad():
        original = model.encode_lemmas([[1, 50, 51, 2]])
        model.summary_slots[1].add_(torch.arange(24) / 10)
        changed_slot = model.encode_lemmas([[1, 50, 51, 2]])
        changed_text = model.encode_lemmas([[1, 70, 71, 2]])
    torch.testing.assert_close(original[:, 0], changed_slot[:, 0], rtol=0, atol=0)
    assert not torch.equal(original[:, 1], changed_slot[:, 1])
    assert not torch.equal(changed_slot, changed_text)


def test_memory_occurrence_order_and_future_memory_does_not_leak():
    model = tiny_model(2)
    ids = torch.tensor([[1, 30, 3, 3, 31, 3, 3, 32, 2]])
    first, second = [1, 50, 2], [1, 80, 81, 2]
    memories = {(0, 5): second, (0, 2): first}  # Deliberately reverse dict order.
    valid = ids.ne(0)
    with torch.no_grad():
        embedded = model._prepare_embeddings(ids, valid, memories)
        vectors = model.encode_lemmas([first, second])
        before = model(ids, memories=memories).logits
        after = model(ids, memories={(0, 2): first, (0, 5): [1, 90, 91, 2]}).logits
        swapped = model(ids, memories={(0, 2): second, (0, 5): first}).logits
    torch.testing.assert_close(embedded[0, 2:4], vectors[0])
    torch.testing.assert_close(embedded[0, 5:7], vectors[1])
    torch.testing.assert_close(before[:, :5], after[:, :5], rtol=0, atol=0)
    assert not torch.equal(before[:, 7], swapped[:, 7])


def test_invalid_or_overlapping_memory_ranges_are_rejected():
    model = tiny_model(2)
    ids = torch.tensor([[1, 3, 3, 3, 2]])
    with pytest.raises(ValueError, match="every MEM"):
        model(ids)
    with pytest.raises(ValueError, match="overlap"):
        model(ids, memories={(0, 1): [50], (0, 2): [51]})
    with pytest.raises(ValueError, match="consecutive"):
        model(torch.tensor([[1, 3, 50, 2]]), memories={(0, 1): [50]})
    with pytest.raises(ValueError, match="outside"):
        model(ids, memories={(0, 4): [50]})
    with pytest.raises(ValueError, match="without MEM"):
        model(torch.tensor([[1, 3, 3, 2]]), memories={(0, 1): [3]})


def test_full_text_ce_shift_and_action_only_labels_match_manual_loss():
    model = tiny_model()
    ids = torch.tensor([[1, 40, 41, 2, 0], [1, 50, 51, 52, 2]])
    output = model(ids, labels=ids)
    targets = ids[:, 1:].clone()
    targets[targets == 0] = -100
    manual = F.cross_entropy(output.logits[:, :-1].reshape(-1, 260), targets.reshape(-1))
    torch.testing.assert_close(output.loss, manual)
    labels = ids.clone()
    labels[:, :3] = -100
    output = model(ids, labels=labels)
    targets = labels[:, 1:].clone()
    targets[targets == 0] = -100
    manual = F.cross_entropy(output.logits[:, :-1].reshape(-1, 260), targets.reshape(-1))
    torch.testing.assert_close(output.loss, manual)
    assert model(ids).loss is None


def test_memory_labels_are_ignored_but_final_slot_predicts_following_text():
    model = tiny_model(2)
    ids = torch.tensor([[1, 40, 3, 3, 41, 2]])
    memory = {(0, 2): [1, 70, 2]}
    output = model(ids, labels=ids, memories=memory)
    targets = torch.tensor([[40, -100, -100, 41, 2]])
    manual = F.cross_entropy(output.logits[:, :-1].reshape(-1, 260), targets.reshape(-1))
    torch.testing.assert_close(output.loss, manual)
    token_log_probs = sequence_log_probs(model, ids, memories=memory, reduction="none")
    assert torch.equal(token_log_probs[:, 1:3], torch.zeros(1, 2))
    torch.testing.assert_close(-token_log_probs.sum() / 3, output.loss)


def test_padding_mask_and_all_ignored_loss_are_finite():
    model = tiny_model()
    short = torch.tensor([[1, 40, 41, 2]])
    right = torch.tensor([[1, 40, 41, 2, 0, 0]])
    left = torch.tensor([[0, 0, 1, 40, 41, 2]])
    with torch.no_grad():
        base = model(short, labels=short)
        padded_right = model(right, labels=right)
        padded_left = model(left, labels=left)
    torch.testing.assert_close(base.logits, padded_right.logits[:, :4])
    torch.testing.assert_close(base.logits, padded_left.logits[:, 2:])
    torch.testing.assert_close(base.loss, padded_right.loss)
    torch.testing.assert_close(base.loss, padded_left.loss)
    all_padding = torch.zeros(1, 4, dtype=torch.long)
    output = model(all_padding, labels=all_padding)
    assert output.loss.item() == 0 and output.logits.isfinite().all()
    output.loss.backward()
    masked = torch.tensor([[1, 40, 80, 90]])
    mask = torch.tensor([[1, 1, 0, 0]])
    loss = model(masked, attention_mask=mask, labels=masked).loss
    logits = model(masked, attention_mask=mask).logits
    torch.testing.assert_close(loss, F.cross_entropy(logits[:, 0], masked[:, 1]))


def test_sequence_log_probs_and_cpu_bfloat16_backward():
    model = tiny_model()
    ids = torch.tensor([[1, 40, 41, 2]])
    log_probs = model.sequence_log_probs(ids, reduction="none")
    torch.testing.assert_close(log_probs.sum(-1), model.sequence_log_probs(ids))
    torch.testing.assert_close(log_probs.mean(-1), model.sequence_log_probs(ids, reduction="mean"))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = model(ids, labels=ids)
    assert output.loss.isfinite()
    output.loss.backward()
    assert torch.isfinite(model.decoder.layers[0].attention.qkv.weight.grad).all()


def test_checkpoint_roundtrip_and_strict_contract():
    model = tiny_model()
    ids = torch.tensor([[1, 40, 3, 3, 41, 2]])
    memories = {(0, 2): [1, 70, 2]}
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "model.pt"
        save_checkpoint(path, model, {"step": 7, "test": True})
        restored, metadata = load_checkpoint(path)
        restored.eval()
        assert metadata == {"step": 7, "test": True}
        with torch.no_grad():
            torch.testing.assert_close(model(ids, memories=memories).logits,
                                       restored(ids, memories=memories).logits, rtol=0, atol=0)
        payload = torch.load(path, weights_only=True)
        assert payload["architecture"] == ARCHITECTURE
        for field, bad in (("architecture", "encoder-decoder"),
                           ("tokenizer_fingerprint", "bad"),
                           ("tokenizer", {})):
            altered = dict(payload)
            altered[field] = bad
            torch.save(altered, path)
            with pytest.raises(ValueError):
                load_checkpoint(path)
        altered = dict(payload)
        altered["config"] = {**payload["config"], "latent_steps": 4}
        torch.save(altered, path)
        with pytest.raises(ValueError, match="configuration"):
            load_checkpoint(path)
        altered = dict(payload)
        altered["model_state"] = dict(payload["model_state"])
        del altered["model_state"]["summary_slots"]
        torch.save(altered, path)
        with pytest.raises(ValueError, match="weights"):
            load_checkpoint(path)


def test_config_and_sequence_limits_are_enforced():
    with pytest.raises(ValueError, match="even"):
        ModelConfig(d_model=21, n_heads=3)
    with pytest.raises(ValueError, match="vocab"):
        ModelConfig(vocab_size=261)
    with pytest.raises(ValueError, match="positive"):
        ModelConfig(lemma_slots=0)
    model = tiny_model()
    with pytest.raises(ValueError, match="max_seq_len"):
        model(torch.ones(1, 65, dtype=torch.long))
    with pytest.raises(ValueError, match="summary slots"):
        model.encode_lemmas([[50] * 64])
