from dataclasses import replace

import pytest

from metamath_generator.parser import MetamathParser
from metamath_generator.verifier import verify
from neural_prover.data import CorpusBuildConfig, ProverExample
from neural_prover.environment import BackwardEnvironment
from neural_prover.tokenizer import MetamathTokenizer
from neural_prover.trajectories import replay_certificate, trajectory_examples


SOURCE = """
$c wff |- P Q R $.
wp $a wff P $.
wq $a wff Q $.
wr $a wff R $.
ap $a |- P $.
ap2 $a |- P $.
aq $a |- Q $.
${
 hp $e |- P $.
 hq $e |- Q $.
 join $a |- R $.
$}
target $p |- R $= ap aq join $.
"""


def fixture():
    db = MetamathParser().parse_text(SOURCE)
    cert = db.proved_theorems['target']
    verify(cert, db)
    env = BackwardEnvironment(db, excluded_assertions=['target'])
    tok = MetamathTokenizer.from_database(db)
    return db, cert, env, tok


def test_certificate_replays_all_goals_and_remaining_cost():
    db, cert, env, tok = fixture()
    trace = replay_certificate(cert, env)
    assert [len(t.before.goals) for t in trace] == [1, 2, 1]
    assert [t.tactic.rule for t in trace] == ['join', 'ap', 'aq']
    assert trace[-1].after.solved
    rows = trajectory_examples(cert, env, tok, CorpusBuildConfig())
    assert [r.value_target for r in rows] == pytest.approx([.97, .98, .99])
    assert [r.remaining_steps for r in rows] == [3, 2, 1]
    for r in rows:
        assert len(tok.proof_state_from_tokens(r.state_tokens, db).goals) == r.goal_count
        assert ProverExample.from_record(r.to_record()) == r


def test_trajectory_token_overflow_does_not_emit_partial_training():
    _, cert, env, tok = fixture()
    with pytest.raises(ValueError, match='token budget'):
        trajectory_examples(cert, env, tok, replace(CorpusBuildConfig(), max_state_tokens=1))


def test_same_state_candidates_include_alternate_verified_actions():
    _, cert, env, tok = fixture()
    rows = trajectory_examples(cert, env, tok, CorpusBuildConfig())
    row = rows[1]  # Both ap and ap2 prove P; the remaining Q must still close.
    positive_tokens = [tok.decode(row.candidate_action_ids[i]) for i in row.positive_candidate_indices]
    assert any('ap' in a for a in positive_tokens)
    assert any('ap2' in a for a in positive_tokens)
    assert row.goal_count == 2


def test_candidate_masks_do_not_mix_other_states_and_multi_positives():
    torch = pytest.importorskip('torch')
    from neural_prover.candidate_training import candidate_batch, candidate_objective
    batch = candidate_batch([(1, 2), (3, 4)],
                            [[(1, 2), (5, 6)], [(3, 4), (7, 8)]], [[0, 1], [0]], 0)
    assert batch['candidate_allowed'].sum(dim=1).tolist() == [2, 2]
    assert batch['candidate_positive'].sum(dim=1).tolist() == [2, 1]

    class Model:
        def score_candidate_matrix(self, state, actions):
            return torch.zeros((2, len(actions)), requires_grad=True), None

    loss, _, _ = candidate_objective(Model(), torch.ones((2, 1)), batch)
    assert loss.item() == pytest.approx(.5 * torch.log(torch.tensor(2.)).item())
    loss.backward()


def tiny_model(tokenizer):
    pytest.importorskip('torch')
    from neural_prover.model import ProofTransformer, ProofTransformerConfig
    return ProofTransformer(ProofTransformerConfig(vocab_size=len(tokenizer), pad_id=tokenizer.pad_id,
        bos_id=tokenizer.bos_id, eos_id=tokenizer.eos_id, d_model=16, nhead=2,
        num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=32,
        max_state_tokens=128, max_action_tokens=128, dropout=0.)).eval()


def test_rlvr_requires_official_gate_and_censors_verifier_failure(monkeypatch):
    pytest.importorskip('torch')
    from neural_prover import rlvr
    from metamath_generator.model import Theorem
    db, _, env, tok = fixture()
    model = tiny_model(tok)
    target = Theorem('test', [], db.statements['ap'].conclusion)
    calls = []

    def reject(*args, **kwargs):
        calls.append(kwargs)
        return False, {'external_verification': {'status': 'error'}}

    monkeypatch.setattr(rlvr, 'verification_record', reject)
    result = rlvr.rollout(model, tok, env, target, rlvr.RLVRConfig(seconds_per_rollout=30), 'unused.mm')
    assert result['reward'] is None
    assert calls[0]['require_external'] is True
    assert rlvr.update_group(model, [result], None, rlvr.RLVRConfig())['reason'] == 'censored_verification'


def test_rlvr_policy_gradient_uses_sampled_action_and_leaves_value_head_alone():
    torch = pytest.importorskip('torch')
    from neural_prover.rlvr import distribution, update_group, RLVRConfig
    _, cert, env, tok = fixture()
    model = tiny_model(tok)
    row = trajectory_examples(cert, env, tok, CorpusBuildConfig())[1]
    actions = list(row.candidate_action_ids)
    assert len(actions) >= 2
    with torch.no_grad():
        old = distribution(model, row.state_ids, actions).tolist()
    decisions = [dict(state=row.state_ids, actions=actions, selected=i, old_log_probs=old) for i in [0, 1]]
    episodes = [dict(reward=float(i == 0), decisions=[d]) for i, d in enumerate(decisions)]
    before = [p.detach().clone() for p in model.value_head.parameters()]
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    assert update_group(model, episodes, optimizer, RLVRConfig())['updated']
    assert all(torch.equal(a, b) for a, b in zip(before, model.value_head.parameters()))
    with torch.no_grad():
        new = distribution(model, row.state_ids, actions).tolist()
    assert new[0] - new[1] > old[0] - old[1]


def test_shortcut_replaces_long_reference_and_relabels_remaining_cost():
    source = SOURCE.replace('target $p', 'ar $a |- R $.\ntarget $p')
    db = MetamathParser().parse_text(source)
    cert = db.proved_theorems['target']
    env = BackwardEnvironment(db, excluded_assertions=['target'])
    tok = MetamathTokenizer.from_database(db)
    rows = trajectory_examples(cert, env, tok, CorpusBuildConfig())
    assert len(rows) == 1
    assert rows[0].reference_steps == 3 and rows[0].remaining_steps == 1
    assert rows[0].rule == 'ar' and rows[0].value_target == pytest.approx(.99)


def test_balanced_weights_protect_rare_structure_without_losing_exploration():
    from neural_prover.sampling import balanced_weights, balanced_order
    common = dict(remaining_steps=1, rule='ap', structure_family='P', hypothesis_count=0, goal_count=1)
    rare = dict(remaining_steps=8, rule='join', structure_family='R', hypothesis_count=2, goal_count=3)
    records = [common] * 99 + [rare]
    weights = balanced_weights(records)
    assert sum(weights) == pytest.approx(1.)
    assert weights[-1] == pytest.approx(.25 / 100 + .75 / 2)
    assert all(w > 0 for w in weights)
    assert rare in balanced_order(records, 17)[:2]


def test_training_sampler_resume_and_validation_distribution(tmp_path):
    pytest.importorskip('torch')
    import gzip
    import json
    from neural_prover.scale_train import _ResumableRecordStream
    rows = [dict(id=str(i), state=[1], action=[2], value=.99, rule=str(i % 2),
                 remaining_steps=i + 1, hypothesis_count=i % 3) for i in range(9)]
    shard = tmp_path / 'data.jsonl.gz'
    with gzip.open(shard, 'wt') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    (tmp_path / 'manifest.json').write_text(json.dumps(dict(training_sampling='balanced',
        shards={s: [dict(path=shard.name)] for s in ['train', 'validation']})))
    whole = _ResumableRecordStream(tmp_path, 'train', 19, 5)
    expected = [next(whole) for _ in range(24)]
    resumed = _ResumableRecordStream(tmp_path, 'train', 19, 5, consumed=7)
    assert [next(resumed) for _ in range(17)] == expected[7:]
    validation = _ResumableRecordStream(tmp_path, 'validation', 19, 5)
    assert sorted(next(validation)['id'] for _ in rows) == [str(i) for i in range(9)]


def test_source_examples_use_same_cost_and_candidate_contract():
    from neural_prover.data import _source_examples
    from neural_prover.audit import _audit_candidates
    db, _, _, tok = fixture()
    env = BackwardEnvironment(db)
    rows = list(_source_examples(db, tok, CorpusBuildConfig()))
    assert rows and all(r.value_target == pytest.approx(.99) for r in rows)
    for row in rows:
        state = tok.proof_state_from_tokens(row.state_tokens, db)
        _audit_candidates(state, row.action_ids, row.candidate_action_ids,
                          row.positive_candidate_indices, tok, env)
    row = rows[0]
    with pytest.raises(ValueError, match='indices'):
        _audit_candidates(state, row.action_ids, row.candidate_action_ids, [999], tok, env)


def test_leakage_index_sees_every_pending_goal_and_scale_shards(tmp_path):
    import gzip
    import json
    from neural_prover.benchmark import _training_leakage_index
    from metamath_generator.quality import semantic_profile
    db, cert, env, tok = fixture()
    row = trajectory_examples(cert, env, tok, CorpusBuildConfig())[1]
    tok.save(tmp_path / 'tokenizer.json')
    (tmp_path / 'train.jsonl').write_text(json.dumps(row.to_record()) + '\n')
    plain = _training_leakage_index((str(tmp_path / 'train.jsonl'),), db)
    assert plain.examples == 1
    assert plain.conclusion_keys == {semantic_profile(db.statements[r]).conclusion_key for r in ('ap', 'aq')}
    scale = tmp_path / 'scale'
    scale.mkdir()
    tok.save(scale / 'tokenizer.json')
    with gzip.open(scale / 'train.gz', 'wt') as f:
        f.write(json.dumps(dict(state=row.state_ids)) + '\n')
    (scale / 'manifest.json').write_text(json.dumps(dict(shards={'train': [{'path': 'train.gz'}]})))
    assert _training_leakage_index((str(scale),), db).conclusion_keys == plain.conclusion_keys


def test_candidate_lengths_participate_in_context_budget():
    pytest.importorskip('torch')
    from neural_prover.scale_train import _record_length
    assert _record_length(dict(state=[1] * 2, action=[2] * 3, candidate_action_ids=[[3] * 12])) == 12


def test_positive_witnesses_merge_only_for_identical_full_states():
    from neural_prover.trajectories import merge_positive_candidates
    _, cert, env, tok = fixture()
    row = trajectory_examples(cert, env, tok, CorpusBuildConfig())[1]
    a, b = row.positive_candidate_indices[:2]
    first = replace(row, positive_candidate_indices=(a,))
    other = replace(row, positive_candidate_indices=(b,), action_ids=row.candidate_action_ids[b])
    different_state = replace(first, state_ids=(*first.state_ids, 0))
    merged = merge_positive_candidates([first, other, different_state])
    assert set(merged[0].positive_candidate_indices) == {a, b}
    assert set(merged[1].positive_candidate_indices) == {a, b}
    assert merged[2].positive_candidate_indices == (a,)
