"""Metamath theorem generation toolkit."""

from .compose import CompositionError, compose
from .database import TheoremDatabase
from .generator import (
    GenerationConfig,
    GenerationSummary,
    TheoremGenerator,
)
from .model import Database, Hypothesis, Node, Proof, Theorem
from .parser import MetamathParser, ParseError, parse
from .quality import (
    QualityAnalyzer,
    QualityAssessment,
    QualityConfig,
    SemanticProfile,
    remove_vacuous_quantifiers,
    semantic_profile,
)
from .unification import (
    OccursCheckError,
    UnificationError,
    substitute,
    unify,
)

__all__ = [
    "CompositionError",
    "Database",
    "GenerationConfig",
    "GenerationSummary",
    "Hypothesis",
    "MetamathParser",
    "Node",
    "OccursCheckError",
    "ParseError",
    "Proof",
    "QualityAnalyzer",
    "QualityAssessment",
    "QualityConfig",
    "SemanticProfile",
    "Theorem",
    "TheoremGenerator",
    "TheoremDatabase",
    "UnificationError",
    "compose",
    "parse",
    "remove_vacuous_quantifiers",
    "semantic_profile",
    "substitute",
    "unify",
]
