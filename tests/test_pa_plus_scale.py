from pathlib import Path
import pytest
from neural_prover.audit import audit_scale_corpus
from neural_prover.data import CorpusBuildConfig, build_corpus
from neural_prover.scale_data import ScaleCorpusConfig, build_scale_corpus, iter_scale_records

ROOT = Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('workers', [1, 2])
def test_pa_plus_scale_templates_replay_with_certified_bridges(tmp_path, workers):
    database = ROOT/'formal/peano-pa-plus.mm'
    base = tmp_path/'base'
    build_corpus(database, base, CorpusBuildConfig(
        seeds=(7,), steps_per_seed=20, definition_catalog=str(ROOT/'formal/pa-plus-definitions.json'),
        bootstrap_definitions=True, bounded_nat_max=2, ground_instances_per_predicate=1,
        max_state_tokens=2304, max_action_tokens=2304,
    ))
    output = tmp_path/'scale'
    build_scale_corpus(database,base,output,ScaleCorpusConfig(
        train_examples=80,validation_examples=16,test_examples=16,shard_size=16,
        workers=workers,kernel_validation_interval=1,
    ))
    records = [record for split in ('train','validation','test') for record in iter_scale_records(output,split)]
    assert any(r['generation_kind']=='definition_bridge' for r in records)
    assert any(r['definition_support'] for r in records)
    audit = audit_scale_corpus(database,output,sample_size=1000)
    assert audit['sample_audited'] == 112
    assert audit['valid_actions'] == 112
    assert audit['invalid_actions'] == 0
