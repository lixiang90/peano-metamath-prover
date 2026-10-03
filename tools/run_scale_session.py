"""Detached, time-bounded RTX training session with durable live telemetry."""
from __future__ import annotations
import argparse
import ctypes
import gc
import gzip
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    # Windows readers can briefly hold a file without delete sharing.
    for attempt in range(100):
        try:
            os.replace(temporary, path)
            break
        except PermissionError:
            if attempt == 99:
                raise
            time.sleep(0.02)

def main():
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument('run_directory', type=Path)
    parser.add_argument('--hours', type=float, default=9)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if not 0 < args.hours <= 12:
        parser.error('hours must be in (0, 12]')
    run = args.run_directory.resolve()
    run.mkdir(parents=True, exist_ok=True)
    (run / 'runner.pid').write_text(str(os.getpid()))
    state = {'status': 'preparing', 'created_at': time.time(), 'pid': os.getpid(),
             'target_seconds': args.hours * 3600, 'message': '正在重新生成并校验训练数据'}
    def update(**values):
        state.update(values)
        state['updated_at'] = time.time()
        write_json(run / 'session.json', state)
        print(json.dumps(values, ensure_ascii=False), flush=True)
    def progress(phase):
        def callback(record):
            now = time.time()
            record.update(timestamp=now, phase=phase)
            with (run / f'{phase}.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(record) + '\n')
            write_json(run / 'live.json', record)
            if record['step'] % 100 == 0 or record['step'] == 1:
                print(f"{phase}: step={record['step']} loss={record['loss']:.5f} elapsed={record['elapsed_seconds']:.1f}s", flush=True)
            if not math.isfinite(record['loss']) or not math.isfinite(record['gradient_norm']):
                state['stop_reason'] = 'nonfinite'
                return True
            if phase == 'training':
                if (run / 'STOP').exists():
                    state['stop_reason'] = 'requested'
                    return True
                if record['elapsed_seconds'] >= args.hours * 3600:
                    state['stop_reason'] = 'time_budget'
                    return True
            return False
        return callback
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        update()
        from monitor_training import process_alive
        while True:
            manifest_path = run / 'corpus/manifest.json'
            if manifest_path.exists():
                try:
                    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
                    if manifest.get('complete'):
                        break
                except json.JSONDecodeError:
                    pass
            err = run / 'prepare.err'
            if err.exists() and 'Traceback (most recent call last)' in err.read_text(encoding='utf-8', errors='replace'):
                raise RuntimeError('Data preparation failed; see prepare.err')
            pid_file = run / 'prepare.pid'
            if pid_file.exists() and not process_alive(int(pid_file.read_text(encoding='utf-8-sig').strip())):
                raise RuntimeError('Data preparation process exited before completion')
            time.sleep(5)
        import torch
        torch.set_num_threads(4)
        if not torch.cuda.is_available() or '3090' not in torch.cuda.get_device_name(0):
            raise RuntimeError('Expected local RTX 3090 CUDA device')
        from neural_prover.audit import audit_scale_corpus, audit_corpus_actions
        from neural_prover.data import load_examples
        from neural_prover.tokenizer import MetamathTokenizer
        from neural_prover.scale_train import ScaleTrainingConfig, train_scale_model, evaluate_scale_checkpoint
        from dataclasses import asdict
        update(status='auditing', message='对新语料抽样执行内核回放校验', gpu=torch.cuda.get_device_name(0))
        tokenizer = MetamathTokenizer.load(run/'corpus/tokenizer.json')
        if not tokenizer.pa_plus_context.enabled or len(tokenizer.pa_plus_context.definition_predicates) != 68:
            raise RuntimeError('Expected all 68 PA+ definitions in tokenizer context')
        base_audit = audit_corpus_actions(ROOT/'formal/peano-pa-plus.mm',run/'base-corpus',run/'base-audit.json')
        if base_audit['invalid_actions']:
            raise RuntimeError('PA+ base corpus failed full action audit')
        # Keep non-augmentable certified ground instances in the training mix.
        ground = [x for x in load_examples(run/'base-corpus/train.jsonl') if x.generation_kind == 'bounded_instance']
        if not ground:
            raise RuntimeError('PA+ training corpus has no bounded instances')
        replay_path = run/'corpus/pa-plus-ground-train.jsonl.gz'
        with gzip.open(replay_path,'wt',encoding='utf-8') as stream:
            for x in ground:
                stream.write(json.dumps(dict(id=x.example_id,state=x.state_ids,action=x.action_ids,value=x.value_target,proof_depth=x.proof_depth,generation_kind=x.generation_kind))+'\n')
        manifest['training_replay'] = dict(path=replay_path.name, records=len(ground), interval=8)
        write_json(run/'corpus/manifest.json',manifest)
        update(message='PA+ 全量基础动作审计通过；正在抽样回放扩增语料',ground_replay_examples=len(ground),formal_system='peano-pa-plus.mm',definitions=68)
        audit = audit_scale_corpus(ROOT/'formal/peano-pa-plus.mm', run/'corpus', run/'audit.json', sample_size=1000)
        if audit['invalid_actions'] or audit['duplicate_ids_in_sample'] or audit['valid_actions'] != 1000:
            raise RuntimeError('Corpus audit did not pass; see audit.json')
        common = dict(micro_batch_size=8,gradient_accumulation_steps=1,device='cuda',
                      learning_rate=1e-4,candidate_loss_weight=0.35,validation_batches=32,checkpoint_every=250,
                      initial_context_tokens=512,context_warmup_steps=500)
        if args.resume:
            cfg = ScaleTrainingConfig(**json.loads((run/'training-config.json').read_text()))
        else:
            update(status='calibrating', message='使用完整 2304-token 上下文测量 RTX 3090 吞吐量')
            pilot = dict(common, max_steps=80, warmup_steps=2, initial_context_tokens=2304,
                         context_warmup_steps=1,checkpoint_every=80,validation_batches=0)
            for micro_batch in (8, 4, 2, 1):
                pilot.update(micro_batch_size=micro_batch, gradient_accumulation_steps=8//micro_batch)
                try:
                    result = train_scale_model(run/'corpus',run/'calibration',ScaleTrainingConfig(**pilot),on_step=progress('calibration'))
                    common.update(micro_batch_size=micro_batch, gradient_accumulation_steps=8//micro_batch)
                    break
                except torch.cuda.OutOfMemoryError:
                    if micro_batch == 1:
                        raise
                    update(message=f'调整显存用量：减小微批量，保持有效批量 8', calibration_micro_batch=micro_batch//2)
                    gc.collect()
                    torch.cuda.empty_cache()

            if state.get('stop_reason'):
                raise RuntimeError('Calibration encountered non-finite values')
            records = result['history']
            seconds_per_step = (records[-2]['elapsed_seconds']-records[9]['elapsed_seconds'])/(records[-2]['step']-records[9]['step'])
            # Include normal checkpoint overhead in the estimate; hard time limit
            # is enforced independently by the callback after each optimizer step.
            max_steps = max(1000, round(args.hours * 3600 / (seconds_per_step + 0.04)))
            cfg = ScaleTrainingConfig(**dict(common,max_steps=max_steps,warmup_steps=max(200,round(max_steps*.02))))
            write_json(run/'training-config.json',asdict(cfg))
            update(calibration_seconds_per_step=seconds_per_step,planned_steps=max_steps)
            del result
            gc.collect()
            torch.cuda.empty_cache()
        update(status='training',message='监督训练进行中 · 检查点每 250 步保存',
               training_started_at=time.time(), configuration=asdict(cfg),planned_steps=cfg.max_steps,
               expected_end_at=time.time()+args.hours*3600)
        result = train_scale_model(run/'corpus',run/'model',cfg,
                                  resume_from=run/'model/latest.pt' if args.resume else None,
                                  on_step=progress('training'))
        reason = state.get('stop_reason')
        if reason == 'nonfinite':
            raise RuntimeError('Training stopped with non-finite loss or gradients; last checkpoint saved')
        checkpoint = Path(result.get('final_checkpoint',result['resume_checkpoint']))
        update(status='evaluating',message='训练已结束，正在执行留出验证集评估',checkpoint=str(checkpoint),stop_reason=reason or 'steps_complete')
        gc.collect()
        torch.cuda.empty_cache()
        evaluation = evaluate_scale_checkpoint(checkpoint,run/'corpus',run/'evaluation-validation.json',split='validation',examples=512,device_name='cuda')
        update(status='complete' if reason != 'requested' else 'stopped',message='训练及验证已完成，权重与续训状态已保存',
               completed_at=time.time(),evaluation=evaluation['metrics'],completed_steps=result['completed_steps'])
    except Exception:
        update(status='failed',message=traceback.format_exc())
        raise
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

if __name__ == '__main__':
    main()
