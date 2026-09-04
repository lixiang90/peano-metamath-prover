"""Shared internal/external certificate gate for evaluation and replay."""
from __future__ import annotations

import math
from pathlib import Path

from .certificate import verify_certificate
from .external import verify_certificate_external


def verification_record(
    certificate,
    database,
    database_path: str | Path,
    *,
    external_verifier: str | None = None,
    require_external: bool = False,
    timeout_seconds: float = 60.0,
) -> tuple[bool, dict]:
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("external timeout must be finite and positive")
    verify_certificate(certificate, database)
    try:
        external = verify_certificate_external(
            certificate, database_path,
            executable=external_verifier, timeout_seconds=timeout_seconds,
        )
        passed = external.passed
        record = external.to_record()
    except Exception as exc:
        passed = False
        record = {"status": "error", "error": str(exc)}
    return (passed if require_external else True), {
        "internal_verified": True,
        "external_verification": record,
        "certification_level": "internal+external" if passed else "internal_only",
    }
