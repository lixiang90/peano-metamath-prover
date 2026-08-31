from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from metamath_generator.database import TheoremDatabase
from metamath_generator.export import export_metamath
from metamath_generator.model import Theorem
from metamath_generator.parser import parse

from .certificate import export_certificate


@dataclass(frozen=True, slots=True)
class ExternalVerificationResult:
    status: str
    verifier: str | None
    executable_sha256: str | None
    returncode: int | None
    elapsed_seconds: float
    output_tail: str
    command: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return self.status == "passed"

    def to_record(self) -> dict:
        record = asdict(self)
        record["command"] = list(self.command)
        return record


def discover_metamath_executable(configured: str | Path | None = None) -> str | None:
    candidates = []
    if configured:
        candidates.append(str(configured))
    environment = os.environ.get("METAMATH_EXECUTABLE")
    if environment:
        candidates.append(environment)
    candidates.extend(("metamath", "metamath-exe", "metamath.exe"))
    for candidate in candidates:
        resolved = shutil.which(candidate)
        if resolved:
            return str(Path(resolved).resolve())
        path = Path(candidate)
        if path.is_file():
            return str(path.resolve())
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _successful_output(output: str) -> bool:
    lowered = output.lower()
    error_markers = (
        "?error",
        "?fatal",
        "proof verification error",
        "there were errors",
    )
    if any(marker in lowered for marker in error_markers):
        return False
    return bool(re.search(
        r"all proofs .* verified|0 errors? (?:were )?found",
        lowered,
    ))


def verify_certificate_external(
    certificate: Theorem,
    database_path: str | Path,
    *,
    executable: str | Path | None = None,
    timeout_seconds: float = 60.0,
) -> ExternalVerificationResult:
    """Replay a certificate with the independent C Metamath executable.

    The source directory is copied to an isolated temporary directory so that
    relative ``$[ ... $]`` includes keep working and no verification artifact
    is written into the formal source tree.
    """

    resolved = discover_metamath_executable(executable)
    if resolved is None:
        return ExternalVerificationResult(
            "not_configured", None, None, None, 0.0, ""
        )
    source = Path(database_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    verifier = Path(resolved)
    if certificate.proof is None or not certificate.proof.source_labels:
        return ExternalVerificationResult(
            "error",
            str(verifier),
            _sha256(verifier) if verifier.is_file() else None,
            None,
            0.0,
            "certificate has no flattened source_labels; "
            "use verify_generated_dag_external for generator DAGs",
            (str(verifier),),
        )
    ambient_database = parse(source)
    import time

    started = time.perf_counter()
    try:
        with tempfile.TemporaryDirectory(prefix="peano-metamath-external-") as raw:
            root = Path(raw) / "formal"
            shutil.copytree(source.parent, root)
            combined = root / f"external-{source.name}"
            fragment = root / "certificate.fragment.mm"
            export_certificate(
                certificate,
                fragment,
                ambient_database=ambient_database,
            )
            combined.write_text(
                (root / source.name).read_text(encoding="utf-8")
                + "\n"
                + fragment.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            command = (str(verifier), str(combined))
            completed = subprocess.run(
                command,
                input="set scroll continuous\nverify proof *\nexit\n",
                text=True,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=root,
                timeout=timeout_seconds,
                check=False,
            )
            output = completed.stdout or ""
            passed = completed.returncode == 0 and _successful_output(output)
            return ExternalVerificationResult(
                "passed" if passed else "failed",
                str(verifier),
                _sha256(verifier),
                completed.returncode,
                time.perf_counter() - started,
                output[-8000:],
                command,
            )
    except (OSError, subprocess.SubprocessError) as exc:
        return ExternalVerificationResult(
            "error",
            str(verifier),
            _sha256(verifier) if verifier.is_file() else None,
            None,
            time.perf_counter() - started,
            str(exc),
            (str(verifier),),
        )


def verify_generated_dag_external(
    theorem: Theorem,
    store: TheoremDatabase,
    database_path: str | Path,
    *,
    executable: str | Path | None = None,
    timeout_seconds: float = 60.0,
) -> ExternalVerificationResult:
    """Export and verify a random-generator proof DAG with C Metamath.

    Generated objects carry parent ids and substitutions, unlike neural-search
    certificates whose proofs are already flattened to ``source_labels``.
    Keeping separate entry points prevents silently exporting an empty proof.
    """

    resolved = discover_metamath_executable(executable)
    if resolved is None:
        return ExternalVerificationResult(
            "not_configured", None, None, None, 0.0, ""
        )
    source = Path(database_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    verifier = Path(resolved)
    import time

    started = time.perf_counter()
    try:
        with tempfile.TemporaryDirectory(
            prefix="peano-metamath-dag-external-"
        ) as raw:
            root = Path(raw) / "formal"
            shutil.copytree(source.parent, root)
            fragment = root / "generated.fragment.mm"
            export_metamath(theorem, store, fragment)
            combined = root / f"external-{source.name}"
            combined.write_text(
                (root / source.name).read_text(encoding="utf-8")
                + "\n"
                + fragment.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            command = (str(verifier), str(combined))
            completed = subprocess.run(
                command,
                input="set scroll continuous\nverify proof *\nexit\n",
                text=True,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=root,
                timeout=timeout_seconds,
                check=False,
            )
            output = completed.stdout or ""
            passed = completed.returncode == 0 and _successful_output(output)
            return ExternalVerificationResult(
                "passed" if passed else "failed",
                str(verifier),
                _sha256(verifier),
                completed.returncode,
                time.perf_counter() - started,
                output[-8000:],
                command,
            )
    except (OSError, subprocess.SubprocessError) as exc:
        return ExternalVerificationResult(
            "error",
            str(verifier),
            _sha256(verifier) if verifier.is_file() else None,
            None,
            time.perf_counter() - started,
            str(exc),
            (str(verifier),),
        )
