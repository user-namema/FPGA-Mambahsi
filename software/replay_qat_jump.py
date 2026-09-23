"""Replay a captured adjacent-epoch validation batch with repeated inference.

Requires only the event directory and matching Python sources, not the dataset.
Native mode restores recorded TF32/cuDNN flags; alternate backend is a diagnosis,
not a promise to reproduce predictions produced on the training GPU.
"""
import argparse
import contextlib
import csv
import json
from pathlib import Path

import numpy as np
import torch

import train_mambahsi_spatial_split_dense_qat as qat
from qat_forensics import (compare_trace, preserve_execution, runtime_metadata,
                           tensor_difference, trace_quantized_batch)


@contextlib.contextmanager
def replay_backend(recorded, mode):
    old_scan = qat.run_selective_scan
    settings = [(torch.backends.cuda.matmul, 'allow_tf32', 'matmul_tf32'),
                (torch.backends.cudnn, 'allow_tf32', 'cudnn_tf32'),
                (torch.backends.cudnn, 'deterministic', 'cudnn_deterministic'),
                (torch.backends.cudnn, 'benchmark', 'cudnn_benchmark')]
    original = [(obj, key, getattr(obj, key)) for obj, key, _ in settings]
    try:
        for obj, key, saved in settings:
            setattr(obj, key, recorded[saved])
        if mode != 'native':
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        if mode == 'reference-tf32-off':
            qat.run_selective_scan = qat._reference_selective_scan
        yield
    finally:
        qat.run_selective_scan = old_scan
        for obj, key, value in original:
            setattr(obj, key, value)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--event-dir', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--mode', choices=['native','tf32-off','reference-tf32-off'], default='native')
    p.add_argument('--save-traces', action='store_true')
    args = p.parse_args(argv)
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        p.error('CUDA unavailable; fix the GPU environment or explicitly request --device cpu')
    root = Path(args.event_dir).expanduser().resolve()
    out = Path(args.output_dir).expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Use a fresh output directory: '+str(out))
    meta = json.loads((root/'event.json').read_text())
    batch = torch.load(root/'validation_batch.pt', map_location='cpu')
    device = torch.device(args.device)
    out.mkdir(parents=True, exist_ok=True)
    report = dict(event_dir=str(root), before_epoch=meta['before_epoch'], after_epoch=meta['after_epoch'],
                  mode=args.mode, device=str(device), scope='captured validation batch only',
                  caveat='First differing node is not proof of a causal defect. Batch metrics are not full-validation OA.',
                  recorded_runtime=meta['runtime'], checkpoints={})
    traces, statistics, outputs = {}, {}, {}
    with preserve_execution(), replay_backend(meta['runtime'], args.mode), torch.no_grad():
        model = qat.build_configured_model(meta['in_channels'], meta['classes'], meta['model_config'])
        qat.prepare_qat_model(model, nbit=meta['nbit'], model_config=meta['model_config'],
            numeric_config=qat.SSMNumericConfig(**meta['numeric_config']), ssm_contract=meta['ssm_contract'])
        model.to(device)
        inputs = batch['inputs'].to(device)
        for name in ('before', 'after'):
            model.load_state_dict(torch.load(root/(name+'.pth'), map_location='cpu'), strict=True)
            qat.freeze_lsq_initialization(model)
            model.eval()
            snapshot = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
            traces[name], statistics[name], outputs[name] = trace_quantized_batch(qat, model, inputs, meta['tile_size'])
            repeat, _, repeat_logits = trace_quantized_batch(qat, model, inputs, meta['tile_size'])
            repeat_rows = compare_trace(traces[name], repeat)
            repeat_changes = [r for r in repeat_rows if r['changed']]
            report['checkpoints'][name] = dict(
                repeat_logits=tensor_difference(outputs[name], repeat_logits),
                repeat_first_difference=repeat_changes[0] if repeat_changes else None,
                repeat_code_changes=sum(r['changed'] for r in repeat_rows if r['kind']=='codes'),
                model_state_unchanged=all(torch.equal(v, model.state_dict()[k].cpu()) for k,v in snapshot.items()),
                captured_batch_metrics=qat.evaluate_prediction(outputs[name].argmax(1).numpy(),
                    batch['labels'].numpy(), meta['classes']))
            expected = batch['expected_'+name+'_prediction']
            valid = batch['labels'] >= 0
            report['checkpoints'][name]['captured_prediction_match'] = float(
                (outputs[name].argmax(1)[valid] == expected[valid]).float().mean())
            del repeat
        report['replay_runtime'] = runtime_metadata(qat)
    rows = compare_trace(traces['before'], traces['after'])
    for row in rows:
        for side in ('before', 'after'):
            for key, value in statistics[side].get(row['trace'], {}).items():
                row[side+'_'+key] = value
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (out/'layer_comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    report['first_code_change'] = next((r for r in rows if r['kind']=='codes' and r['changed']), None)
    report['first_activation_code_change'] = next((r for r in rows if r['kind']=='codes' and r['changed']
        and r.get('before_group') in ('activation','dt_input','dt_output')), None)
    report['dense_logits_difference'] = tensor_difference(outputs['before'], outputs['after'])
    report['source_hash_match'] = meta['runtime']['source_sha256'] == report['replay_runtime']['source_sha256']
    labels = batch['labels'].numpy()
    before, after = [outputs[k].argmax(1).numpy() for k in ('before','after')]
    valid = labels >= 0
    report['prediction_exchange'] = dict(labeled=int(valid.sum()),
        before_correct_after_wrong=int(((before==labels)&(after!=labels)&valid).sum()),
        before_wrong_after_correct=int(((before!=labels)&(after==labels)&valid).sum()),
        prediction_changes=int(((before!=after)&valid).sum()))
    np.savez_compressed(out/'batch_logits.npz', before=outputs['before'].numpy(),
                        after=outputs['after'].numpy(), labels=labels)
    if args.save_traces:
        torch.save(traces, out/'traces.pt')
    qat.save_json(out/'jump_replay.json', report)
    print(json.dumps(report['prediction_exchange']))
    for name, row in report['checkpoints'].items():
        print(f"{name}: saved validation prediction match={row['captured_prediction_match']:.8f}; "
              f"repeat logit changes={row['repeat_logits']['changed']}")
    if not report['source_hash_match']:
        print('WARNING: source hashes differ from capture; inspect runtime metadata before attributing differences.')
    print('Replay report: '+str(out/'jump_replay.json'))


if __name__ == '__main__':
    main()
