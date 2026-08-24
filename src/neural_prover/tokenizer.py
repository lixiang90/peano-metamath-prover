from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator

from metamath_generator.model import (
    Database,
    Hypothesis,
    Node,
    Theorem,
    normalized_pair,
)
from metamath_generator.parser import MetamathParser, ParseError


INSTRUCTION_TOKENS = (
    "<PAD>",
    "<UNK>",
    "<BOS>",
    "<EOS>",
    "<STATE>",
    "<GOAL>",
    "<HYPOTHESES>",
    "<NO_HYP>",
    "<HYP>",
    "<END_HYP>",
    "<DV>",
    "<DV_PAIR>",
    "<END_STATE>",
    "<ACTION>",
    "<APPLY>",
    "<ASSUMPTION>",
    "<RULE>",
    "<SUBST>",
    "<BIND>",
    "<TO>",
    "<END_BIND>",
    "<END_ACTION>",
    "<VALUE>",
    "<SEP>",
    "<PROPOSE_LEMMA>",
    "<LEMMA>",
    "<END_LEMMA>",
)
VARIABLE_TOKEN_PATTERN = re.compile(r"^<V:(.+):(\d+)>$")


@dataclass(frozen=True, slots=True)
class TokenizerConfig:
    max_variables_per_type: int = 32
    strict: bool = True


@dataclass(slots=True)
class CanonicalVariables:
    """Stable theorem-local variable names shared by states and actions."""

    variable_types: dict[str, str]
    max_variables_per_type: int
    actual_to_token: dict[str, str]
    token_to_actual: dict[str, str]

    @classmethod
    def from_theorem(
        cls,
        theorem: Theorem,
        max_variables_per_type: int,
    ) -> "CanonicalVariables":
        counters: dict[str, int] = {}
        actual_to_token: dict[str, str] = {}
        token_to_actual: dict[str, str] = {}

        def register(variable: str) -> None:
            typecode = theorem.variable_types.get(variable)
            if typecode is None or variable in actual_to_token:
                return
            index = counters.get(typecode, 0)
            if index >= max_variables_per_type:
                raise ValueError(
                    f"too many {typecode} variables for tokenizer: "
                    f"{index + 1} > {max_variables_per_type}"
                )
            token = variable_token(typecode, index)
            counters[typecode] = index + 1
            actual_to_token[variable] = token
            token_to_actual[token] = variable

        expressions = [
            *(hypothesis.expr for hypothesis in theorem.hypotheses),
            theorem.conclusion,
        ]
        for expression in expressions:
            for node in expression.walk():
                register(node.op)
        for left, right in sorted(theorem.d_constraints):
            register(left)
            register(right)
        # Defensive support for mandatory variables not occurring in the
        # pretty-printed theorem.
        for variable in sorted(theorem.variable_types):
            register(variable)
        return cls(
            variable_types=theorem.variable_types,
            max_variables_per_type=max_variables_per_type,
            actual_to_token=actual_to_token,
            token_to_actual=token_to_actual,
        )

    def encode_symbol(self, symbol: str) -> str:
        return self.actual_to_token.get(symbol, symbol)

    def decode_symbol(self, token: str) -> str:
        return self.token_to_actual.get(token, token)


def variable_token(typecode: str, index: int) -> str:
    return f"<V:{typecode}:{index}>"


class MetamathTokenizer:
    """A lossless symbol tokenizer for Metamath states and tactics.

    There is no BPE, byte fallback, or natural-language vocabulary.  Every
    formal symbol and every control instruction is an atomic token.
    """

    def __init__(
        self,
        tokens: Iterable[str],
        config: TokenizerConfig | None = None,
        *,
        preserve_token_order: bool = False,
    ) -> None:
        self.config = config or TokenizerConfig()
        supplied = list(dict.fromkeys(tokens))
        if preserve_token_order:
            # Serialized vocabularies define checkpoint-facing token IDs.
            # Never insert newly introduced control tokens into an old file:
            # an explicit upgrade appends them without shifting existing IDs.
            ordered = supplied
        else:
            ordered = [
                *INSTRUCTION_TOKENS,
                *(
                    token for token in supplied
                    if token not in INSTRUCTION_TOKENS
                ),
            ]
        self.tokens = tuple(ordered)
        self.token_to_id = {
            token: index for index, token in enumerate(self.tokens)
        }
        self.id_to_token = dict(enumerate(self.tokens))

    @classmethod
    def from_database(
        cls,
        database: Database,
        config: TokenizerConfig | None = None,
    ) -> "MetamathTokenizer":
        cfg = config or TokenizerConfig()
        formal = set(database.symbols)
        formal.update(database.variables)
        formal.update(database.types)
        formal.update(database.statements)
        typecodes = set(database.types)
        typecodes.update(database.variable_types.values())
        for theorem in database.statements.values():
            typecodes.update(theorem.variable_types.values())
        slots = {
            variable_token(typecode, index)
            for typecode in typecodes
            for index in range(cfg.max_variables_per_type)
        }
        return cls(
            [
                *INSTRUCTION_TOKENS,
                *sorted(formal),
                *sorted(slots),
            ],
            cfg,
        )

    def __len__(self) -> int:
        return len(self.tokens)

    @property
    def pad_id(self) -> int:
        return self.token_to_id["<PAD>"]

    @property
    def bos_id(self) -> int:
        return self.token_to_id["<BOS>"]

    @property
    def eos_id(self) -> int:
        return self.token_to_id["<EOS>"]

    def encode(self, tokens: Iterable[str]) -> list[int]:
        result: list[int] = []
        for token in tokens:
            token_id = self.token_to_id.get(token)
            if token_id is None:
                if self.config.strict:
                    raise ValueError(f"token is outside formal vocabulary: {token}")
                token_id = self.token_to_id["<UNK>"]
            result.append(token_id)
        return result

    def decode(self, token_ids: Iterable[int]) -> list[str]:
        return [self.id_to_token[int(token_id)] for token_id in token_ids]

    def canonical_variables(self, theorem: Theorem) -> CanonicalVariables:
        return CanonicalVariables.from_theorem(
            theorem,
            self.config.max_variables_per_type,
        )

    @staticmethod
    def _node_tokens(
        node: Node,
        variables: CanonicalVariables,
    ) -> Iterator[str]:
        yield variables.encode_symbol(node.op)
        for argument in node.args:
            yield from MetamathTokenizer._node_tokens(argument, variables)

    def state_tokens(
        self,
        theorem: Theorem,
        variables: CanonicalVariables | None = None,
    ) -> list[str]:
        canonical = variables or self.canonical_variables(theorem)
        tokens = ["<BOS>", "<STATE>", "<HYPOTHESES>"]
        if theorem.hypotheses:
            for hypothesis in theorem.hypotheses:
                tokens.append("<HYP>")
                tokens.extend(self._node_tokens(hypothesis.expr, canonical))
                tokens.append("<END_HYP>")
        else:
            tokens.append("<NO_HYP>")
        tokens.append("<GOAL>")
        tokens.extend(self._node_tokens(theorem.conclusion, canonical))
        tokens.append("<DV>")
        for left, right in sorted(theorem.d_constraints):
            tokens.extend([
                "<DV_PAIR>",
                canonical.encode_symbol(left),
                canonical.encode_symbol(right),
            ])
        tokens.extend(["<END_STATE>", "<EOS>"])
        return tokens

    def action_tokens(
        self,
        theorem: Theorem,
        database: Database,
        variables: CanonicalVariables | None = None,
    ) -> list[str]:
        if theorem.proof is None:
            raise ValueError(f"{theorem.name} has no generated proof action")
        rule = database.statements.get(theorem.proof.rule)
        if rule is None:
            raise ValueError(f"unknown proof rule {theorem.proof.rule!r}")
        canonical = variables or self.canonical_variables(theorem)
        substitution: dict[str, Node] = {}
        floating_order = [
            floating.expr.args[0].op for floating in rule.floating
        ]
        if not floating_order:
            floating_order = list(rule.variable_types)
        for variable in floating_order:
            key = f"__rule_{variable}"
            value = theorem.proof.substitution.get(key)
            if value is not None and value != Node(key):
                substitution[variable] = value
        return self.tactic_tokens(
            rule.name,
            substitution,
            canonical,
            variable_order=floating_order,
        )

    def tactic_tokens(
        self,
        rule_name: str,
        substitution: dict[str, Node],
        variables: CanonicalVariables,
        *,
        variable_order: Iterable[str] | None = None,
    ) -> list[str]:
        tokens = [
            "<BOS>",
            "<ACTION>",
            "<APPLY>",
            "<RULE>",
            rule_name,
            "<SUBST>",
        ]
        order = list(variable_order or sorted(substitution))
        for variable in order:
            value = substitution.get(variable)
            if value is None:
                continue
            tokens.extend(["<BIND>", variable, "<TO>"])
            tokens.extend(self._node_tokens(value, variables))
            tokens.append("<END_BIND>")
        tokens.extend(["<END_ACTION>", "<EOS>"])
        # Strictly prove that the vocabulary covers this complete action.
        self.encode(tokens)
        return tokens

    def lemma_tactic_tokens(
        self,
        lemma: Node,
        variables: CanonicalVariables,
    ) -> list[str]:
        """Encode a kernel-checked intermediate-lemma proposal."""

        if lemma.op != "|-" or len(lemma.args) != 1:
            raise ValueError("an intermediate lemma must be a |- assertion")
        tokens = [
            "<BOS>",
            "<ACTION>",
            "<PROPOSE_LEMMA>",
            "<LEMMA>",
        ]
        tokens.extend(self._node_tokens(lemma, variables))
        tokens.extend(["<END_LEMMA>", "<END_ACTION>", "<EOS>"])
        self.encode(tokens)
        return tokens

    @property
    def supports_lemma_actions(self) -> bool:
        return all(
            token in self.token_to_id
            for token in (
                "<PROPOSE_LEMMA>", "<LEMMA>", "<END_LEMMA>"
            )
        )

    def upgraded_for_lemma_actions(self) -> "MetamathTokenizer":
        """Append lemma controls while preserving every existing token ID."""

        tokens = [*self.tokens]
        tokens.extend(
            token for token in (
                "<PROPOSE_LEMMA>", "<LEMMA>", "<END_LEMMA>"
            )
            if token not in self.token_to_id
        )
        return MetamathTokenizer(
            tokens,
            self.config,
            preserve_token_order=True,
        )

    def theorem_from_state_tokens(
        self,
        tokens: Iterable[str],
        database: Database,
        *,
        name: str = "_decoded_state",
    ) -> Theorem:
        """Decode the lossless formal state format back into a theorem."""

        sequence = list(tokens)
        variable_types = {
            token: match.group(1)
            for token in sequence
            for match in [VARIABLE_TOKEN_PATTERN.match(token)]
            if match is not None
        }
        parser = MetamathParser()
        parser.database = database
        old_types = dict(database.variable_types)
        database.variable_types.update(variable_types)
        hypotheses: list[Hypothesis] = []
        d_constraints: set[tuple[str, str]] = set()
        try:
            cursor = sequence.index("<HYPOTHESES>") + 1
            while cursor < len(sequence) and sequence[cursor] != "<GOAL>":
                if sequence[cursor] == "<NO_HYP>":
                    cursor += 1
                    continue
                if sequence[cursor] != "<HYP>":
                    raise ValueError("malformed HYPOTHESES section")
                end = sequence.index("<END_HYP>", cursor + 1)
                expression = parser.parse_expression(
                    sequence[cursor + 1:end]
                )
                hypotheses.append(Hypothesis(
                    f"{name}_h{len(hypotheses)}",
                    expression,
                ))
                cursor = end + 1
            if cursor >= len(sequence) or sequence[cursor] != "<GOAL>":
                raise ValueError("state has no GOAL section")
            goal_end = sequence.index("<DV>", cursor + 1)
            conclusion = parser.parse_expression(
                sequence[cursor + 1:goal_end]
            )
            cursor = goal_end + 1
            while (
                cursor < len(sequence)
                and sequence[cursor] != "<END_STATE>"
            ):
                if (
                    sequence[cursor] != "<DV_PAIR>"
                    or cursor + 2 >= len(sequence)
                ):
                    raise ValueError("malformed DV section")
                d_constraints.add(normalized_pair(
                    sequence[cursor + 1],
                    sequence[cursor + 2],
                ))
                cursor += 3
        except (ValueError, ParseError) as exc:
            raise ValueError(f"cannot decode proof state: {exc}") from exc
        finally:
            database.variable_types.clear()
            database.variable_types.update(old_types)
        return Theorem(
            name,
            hypotheses,
            conclusion,
            d_constraints=d_constraints,
            variable_types=variable_types,
            kind="proof_state",
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(
                {
                    "format": "peano-metamath-tokenizer-v1",
                    "config": asdict(self.config),
                    "tokens": list(self.tokens),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "MetamathTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format") != "peano-metamath-tokenizer-v1":
            raise ValueError("unsupported tokenizer format")
        return cls(
            payload["tokens"],
            TokenizerConfig(**payload["config"]),
            preserve_token_order=True,
        )
