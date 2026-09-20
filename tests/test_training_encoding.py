from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from metamath_generator.model import Node
from metamath_generator.parser import parse
from metamath_generator.unification import substitute_simultaneous
from metamath_generator.definitions import load_definition_catalog
from neural_prover.environment import (
    BackwardEnvironment, ProofState, Tactic, LEMMA_COMMIT_OP, parse_tactic_tokens,
)
from neural_prover.tokenizer import MetamathTokenizer

ROOT = Path(__file__).resolve().parents[1]


class CompleteStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.database = parse(ROOT / 'formal/peano.mm')
        cls.tokenizer = MetamathTokenizer.from_database(cls.database)

    def state(self, goals):
        return ProofState((), tuple(goals), frozenset(), (('x', 'var'), ('y', 'var'), ('z', 'var')))

    def test_complete_queue_round_trip_and_order(self):
        a = self.database.statements['pa_ax1'].conclusion
        b = substitute_simultaneous(a, {'x': Node('z')})
        # Give b and c different structure so swapping them is not alpha renaming.
        c = Node('|-', (Node('=', (Node('y'), Node('0'))),))
        states = [self.state((a,)), self.state((a, b)), self.state((a, b, c)),
                  self.state((a, c, b)), self.state(())]
        encoded = [self.tokenizer.proof_state_tokens(s) for s in states]
        self.assertEqual(len({tuple(x) for x in encoded}), len(states))
        self.assertEqual(encoded[0], self.tokenizer.state_tokens(states[0].as_theorem()))
        for tokens in encoded:
            self.tokenizer.encode(tokens)
            decoded = self.tokenizer.proof_state_from_tokens(tokens, self.database)
            self.assertEqual(self.tokenizer.proof_state_tokens(decoded), tokens)
        with self.assertRaisesRegex(ValueError, 'one logical goal'):
            self.tokenizer.theorem_from_state_tokens(encoded[1], self.database)

    def test_pending_lemma_is_not_an_active_hypothesis_or_ordinary_goal(self):
        a = self.database.statements['pa_ax1'].conclusion
        marker = Node(LEMMA_COMMIT_OP, (a,))
        state = self.state((a, marker, a))
        tokens = self.tokenizer.proof_state_tokens(state)
        self.assertNotIn(LEMMA_COMMIT_OP, tokens)
        self.tokenizer.encode(tokens)
        decoded = self.tokenizer.proof_state_from_tokens(tokens, self.database)
        self.assertFalse(decoded.hypotheses)
        self.assertEqual(decoded.goals[1].op, LEMMA_COMMIT_OP)
        self.assertEqual(self.tokenizer.proof_state_tokens(decoded), tokens)
        self.assertNotEqual(tokens, self.tokenizer.proof_state_tokens(self.state((a, a, a))))


    def test_action_round_trip_uses_variables_from_all_goals(self):
        def equality(v):
            return Node("=", (Node(v), Node(v)))
        state = self.state(Node("|-", (equality(v),)) for v in ("x", "z", "y"))
        rule = self.database.statements["ax-mp"]
        tactic = Tactic.create(rule.name, {"phi": equality("y"), "psi": equality("x")})
        tokens = self.tokenizer.assertion_tactic_tokens(
            rule, tactic.substitution_dict(), self.tokenizer.proof_state_variables(state),
        )
        self.assertEqual(parse_tactic_tokens(tokens, state, self.tokenizer, self.database), tactic)
        decoded = self.tokenizer.proof_state_from_tokens(
            self.tokenizer.proof_state_tokens(state), self.database,
        )
        restored = parse_tactic_tokens(tokens, decoded, self.tokenizer, self.database)
        self.assertEqual(self.tokenizer.assertion_tactic_tokens(
            rule, restored.substitution_dict(), self.tokenizer.proof_state_variables(decoded),
        ), tokens)


try:
    import torch
    from neural_prover.rl import ReplayBuffer
    from neural_prover.scale_data import _action_tokens
    from neural_prover.search import TransformerPolicy
    from neural_prover.model import ProofTransformer, ProofTransformerConfig
    from neural_prover.mcts import MCTSExperience
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'PyTorch required')
class TrainingEncodingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.database = parse(ROOT / 'formal/peano-pa-plus.mm')
        catalog = load_definition_catalog(ROOT / 'formal/pa-plus-definitions.json')
        cls.tokenizer = MetamathTokenizer.from_database(
            cls.database, pa_plus_context=MetamathTokenizer.pa_plus_context_from_database(
                cls.database, catalog.definition_names, catalog.statement_names,
            ),
        )
        cls.environment = BackwardEnvironment(cls.database)
        cls.environment.configure_from_tokenizer(cls.tokenizer)

    def test_scale_replay_and_inference_share_typed_bindings(self):
        for name in ('pa_ax1', 'gen_df_positive_fold'):
            assertion = self.environment.assertions[name]
            state = ProofState.from_theorem(assertion)
            tactic = Tactic.create(name, {v: Node(v) for v in assertion.variable_types})
            tokens = _action_tokens(tactic, assertion, self.tokenizer, self.environment)
            self.assertEqual(tokens, ReplayBuffer._tactic_tokens(tactic, state, self.tokenizer, self.environment))
            self.assertTrue(all(tokens[i + 1].startswith('<V:') for i, t in enumerate(tokens) if t == '<BIND>'))
            restored = parse_tactic_tokens(tokens, assertion, self.tokenizer, self.database, environment=self.environment)
            self.assertEqual(restored, tactic)
            experience = MCTSExperience(state, (tactic,), (1.0,), 1.0, True, True)
            replay = ReplayBuffer()
            replay.add_mcts(SimpleNamespace(experiences=(experience,)), self.tokenizer, self.environment)
            self.assertEqual(replay.examples[0].policy_weight, 1.0)
            self.assertEqual(replay.examples[0].action_ids, (tuple(self.tokenizer.encode(tokens)),))

    def test_replay_keeps_same_first_goal_with_different_obligations(self):
        assertion = self.database.statements['pa_ax1']
        state = ProofState.from_theorem(assertion)
        other = ProofState(state.hypotheses, (*state.goals, state.current_goal), state.d_constraints, state.variable_types)
        tactic = Tactic.create(assertion.name, {'x': Node('x')})
        experiences = tuple(MCTSExperience(s, (tactic,), (1.0,), value, True, True)
                            for s, value in ((state, 0.99), (other, 0.98)))
        replay = ReplayBuffer()
        replay.add_mcts(SimpleNamespace(experiences=experiences), self.tokenizer, self.environment)
        self.assertEqual(len(replay), 2)
        self.assertNotEqual(replay.examples[0].state_ids, replay.examples[1].state_ids)
        self.assertEqual([e.value_target for e in replay.examples], [0.99, 0.98])

    def test_inference_encodes_full_current_and_next_states(self):
        assertion = self.database.statements['pa_ax1']
        initial = ProofState.from_theorem(assertion)
        state = ProofState(initial.hypotheses, initial.goals * 3, initial.d_constraints, initial.variable_types)
        tactic = Tactic.create(assertion.name, {'x': Node('x')})
        model = ProofTransformer(ProofTransformerConfig(
            vocab_size=len(self.tokenizer), pad_id=self.tokenizer.pad_id,
            bos_id=self.tokenizer.bos_id, eos_id=self.tokenizer.eos_id,
            d_model=16, nhead=2, num_encoder_layers=1, num_decoder_layers=1,
            dim_feedforward=32, dropout=0.0,
        ))
        model.candidate_head_trained = True
        policy = TransformerPolicy(model, self.tokenizer, self.environment, 'cpu')
        with patch.object(model, 'encode', wraps=model.encode) as encode, patch.object(
            model, 'score_candidates_from_memory', wraps=model.score_candidates_from_memory,
        ) as score:
            ranked = policy.rank(state, [tactic])
        self.assertEqual(len(ranked), 1)
        states = [call.args[0][0].tolist() for call in encode.call_args_list]
        next_state = self.environment.apply(state, tactic).after
        self.assertEqual(states, [self.tokenizer.encode(self.tokenizer.proof_state_tokens(s))
                                  for s in (state, next_state)])
        self.assertEqual(score.call_args.args[0][0].tolist(), self.tokenizer.encode(
            ReplayBuffer._tactic_tokens(tactic, state, self.tokenizer, self.environment),
        ))


    def test_legacy_vocabulary_falls_back_without_dropping_pending_lemma(self):
        from neural_prover.lemma import LemmaBackwardEnvironment

        legacy = MetamathTokenizer(
            [t for t in self.tokenizer.tokens if t not in {"<LEMMA>", "<END_LEMMA>", "<PROPOSE_LEMMA>"}],
            preserve_token_order=True, pa_plus_context=self.tokenizer.pa_plus_context,
        )
        environment = LemmaBackwardEnvironment(self.database)
        assertion = self.database.statements["pa_ax1"]
        initial = ProofState.from_theorem(assertion)
        marker = Node(LEMMA_COMMIT_OP, (initial.current_goal,))
        state = ProofState((), (initial.current_goal, marker, initial.current_goal),
                           initial.d_constraints, initial.variable_types)
        tactic = Tactic.create(assertion.name, {"x": Node("x")})
        model = ProofTransformer(ProofTransformerConfig(
            vocab_size=len(legacy), pad_id=legacy.pad_id, bos_id=legacy.bos_id,
            eos_id=legacy.eos_id, d_model=16, nhead=2, num_encoder_layers=1,
            num_decoder_layers=1, dim_feedforward=32, dropout=0.0,
        ))
        policy = TransformerPolicy(model, legacy, environment, "cpu")
        with patch.object(model, "encode", side_effect=AssertionError("legacy state reached neural encoder")):
            self.assertEqual(len(policy.rank(state, [tactic])), 1)
        replay = ReplayBuffer()
        experience = MCTSExperience(state, (tactic,), (1.0,), 1.0, True, True)
        replay.add_mcts(SimpleNamespace(experiences=(experience,)), legacy, environment)
        self.assertEqual(len(replay), 0)
