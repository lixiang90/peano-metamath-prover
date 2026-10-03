import hashlib
import json
from pathlib import Path

import pytest

from pa_prover_v2.cli import main
from pa_prover_v2.data import DataConfig, generate_corpus

ROOT = Path(__file__).resolve().parents[2]


def test_audit_rejects_unexecuted_actions_after_finish(tmp_path):
    database = ROOT / "formal/peano-pa-plus.mm"
    generate_corpus(database, tmp_path, DataConfig(seeds=(7,), steps_per_seed=10, max_examples=8))
    main(["audit", str(database), str(tmp_path)])
    path = tmp_path / "train.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    records[0]["actions"].append({"op": "APPLY", "args": {"rule": "fake-after-finish"}})
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="terminal"):
        main(["audit", str(database), str(tmp_path)])
