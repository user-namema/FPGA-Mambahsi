"""Trace one identical test tile alone and in its saved evaluation batch.

No calibration, training or checkpoint writes. Trace tensors are selected by
tile before moving to CPU. Flattened spatial/spectral batches retain tile order.
"""
import argparse
import contextlib
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import train_mambahsi_spatial_split_dense_qat as qat
import both_FPGA_single_qat_source as sim


@contextlib.contextmanager
def backend_mode(mode):
    original = qat.run_selective_scan
    matmul = torch.backends.cuda.matmul.allow_tf32
    cudnn = torch.backends.cudnn.allow_tf32
    try:
        if mode != 'native':
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        if mode == 'reference-tf32-off':
            qat.run_selective_scan = qat._reference_selective_scan
        yield
    finally:
        qat.run_selective_scan = original
        torch.backends.cuda.matmul.allow_tf32 = matmul
        torch.backends.cudnn.allow_tf32 = cudnn


def backend_description():
    return dict(matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_tf32=torch.backends.cudnn.allow_tf32,
                cudnn_deterministic=torch.backends.cudnn.deterministic,
                cudnn_benchmark=torch.backends.cudnn.benchmark,
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled())


@torch.no_grad()
def trace_tile(model, inputs, target=0):
    """Return execution-ordered target-tile traces, restoring hooks on error."""
    batch = inputs.shape[0]
    if target < 0 or target >= batch:
        raise ValueError('target outside batch')
    traces, handles, core_stack, counts = {}, [], [], {}
    original_scan = qat.run_selective_scan

    def save(name, value, kind='float'):
        if not isinstance(value, torch.Tensor):
            return
        if value.ndim == 0 or value.shape[0] % batch:
            raise ValueError('Cannot map trace to independent tiles: ' + name)
        selected = value.detach().reshape(batch, -1)[target].cpu().clone()
        if not torch.isfinite(selected).all():
            raise FloatingPointError('Nonfinite trace: ' + name)
        index = counts.get(name, 0); counts[name] = index + 1
        traces[f'{name}#{index}'] = (kind, selected)

    def hook(name):
        def collect(module, args, result):
            if isinstance(module, qat.LsqQuantizer4input):
                save(name + '/input', args[0])
                save(name + '/codes', sim.round_half_away_from_zero(result[0] / module.s).to(torch.int64), 'codes')
                save(name + '/output', result[0])
            else:
                save(name + '/output', result)
        return collect

    def enter(name):
        def pre(module, args):
            core_stack.append(name)
        return pre

    def leave(module, args, result):
        core_stack.pop()

    def scan(u, delta, A, B, C, D=None):
        name = core_stack[-1] if core_stack else 'unknown_core'
        for label, tensor in [('u',u),('dt',delta),('B',B),('C',C)]:
            save(name + '/scan_' + label, tensor)
        result = original_scan(u, delta, A, B, C, D=D)
        save(name + '/scan_output', result)
        return result

    types = (qat.LsqQuantizer4input, nn.Linear, nn.Conv1d, nn.Conv2d,
             nn.BatchNorm1d, nn.BatchNorm2d, nn.ReLU, nn.SiLU,
             nn.AvgPool2d, qat.CurrentMambaCore, qat.SpaMamba, qat.SpeMamba)
    try:
        for name, module in model.named_modules():
            if isinstance(module, qat.CurrentMambaCore):
                handles.append(module.register_forward_pre_hook(enter(name)))
            if isinstance(module, types):
                handles.append(module.register_forward_hook(hook(name)))
            if isinstance(module, qat.CurrentMambaCore):
                handles.append(module.register_forward_hook(leave))
        qat.run_selective_scan = scan
        output = model(inputs)
        output = torch.nn.functional.interpolate(output, size=(16,16), mode='bilinear', align_corners=True)
        save('dense_logits', output)
        logits = output[target].detach().cpu().clone()
    finally:
        qat.run_selective_scan = original_scan
        for handle in handles:
            handle.remove()
    return traces, logits


def compare_traces(reference, actual, comparison, atol=1e-6, rtol=1e-5):
    if list(reference) != list(actual):
        raise ValueError('Execution order/trace keys differ between compared runs')
    rows = []
    for order, (name, (kind, ref)) in enumerate(reference.items()):
        other_kind, act = actual[name]
        if kind != other_kind or ref.shape != act.shape:
            raise ValueError('Trace shape/type mismatch: ' + name)
        diff = (ref.double() - act.double()).abs()
        changed = ref != act
        outside = changed if kind == 'codes' else diff > atol + rtol * ref.double().abs()
        rows.append(dict(comparison=comparison, order=order, trace=name, kind=kind,
            count=ref.numel(), exact_mismatches=int(changed.sum()),
            tolerance_mismatches=int(outside.sum()), mae=float(diff.mean()),
            max_abs_error=float(diff.max())))
    return rows


def first_differences(rows):
    def first(predicate):
        return next((r for r in rows if predicate(r)), None)
    return dict(first_exact=first(lambda r:r['exact_mismatches']>0),
                first_above_tolerance=first(lambda r:r['tolerance_mismatches']>0),
                first_code_change=first(lambda r:r['kind']=='codes' and r['exact_mismatches']>0))


def select_mismatch_group(one, many, test_indices, blocks, batch_size):
    """Ignore background/other splits when choosing a representative tile."""
    mismatch=np.zeros(one.size,dtype=bool)
    mismatch[test_indices]=one.ravel()[test_indices]!=many.ravel()[test_indices]
    mismatch=mismatch.reshape(one.shape)
    found=next((i for i,b in enumerate(blocks)
                if mismatch[b['top']:b['bottom'],b['left']:b['right']].any()),None)
    if found is None:
        raise ValueError('No batch-dependent labeled test predictions in this directory; choose --group-index for numerical-only diagnosis')
    return divmod(found,batch_size)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--qat-run-dir', required=True)
    p.add_argument('--fp32-dir')
    p.add_argument('--data-path', required=True)
    p.add_argument('--device', choices=['cpu','cuda'], default='cuda')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--group-index', type=int, help='Zero-based group in saved test evaluation order; default first group')
    p.add_argument('--target-index', type=int, default=0, help='Tile position within that group')
    p.add_argument('--prediction-dir', help='Existing simulator output: automatically choose a tile with batch-dependent test predictions')
    p.add_argument('--modes', nargs='+', choices=['native','tf32-off','reference-tf32-off'],
                   default=['native','tf32-off','reference-tf32-off'])
    p.add_argument('--atol', type=float, default=1e-6)
    p.add_argument('--rtol', type=float, default=1e-5)
    p.add_argument('--save-traces', action='store_true', help='Save native single/batch vectors for changed stages only')
    args = p.parse_args(argv)
    if (args.group_index is not None and args.group_index < 0) or args.target_index < 0 or args.atol < 0 or args.rtol < 0:
        p.error('Indices and tolerances must be nonnegative')
    if args.prediction_dir and (args.group_index is not None or args.target_index != 0):
        p.error('--prediction-dir selects the group/target automatically; omit manual indices')
    out = Path(args.output_dir).expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Use an empty output directory: ' + str(out))
    run = Path(args.qat_run_dir).expanduser().resolve()
    meta = sim.load_json_object(run/'result.json')
    fp32 = Path(args.fp32_dir).expanduser().resolve() if args.fp32_dir else sim.infer_fp32_dir(meta,run)
    cfg = dict(qat.DEFAULT_MODEL_CONFIG); cfg.update(meta.get('model_config',{}))
    if not qat.d_path_simulator_compatible(cfg):
        raise ValueError('Batch tracer requires supported current D0/D1 topology')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; repair the server driver or explicitly select cpu')
    raw, gt, _ = qat.load_dataset(meta['dataset'], args.data_path)
    raw, gt, classes = sim.validate_raw_data(raw, gt, meta['dataset'])
    image = sim.transform_with_saved_preprocess(raw,fp32/'train_only_preprocess.npz')
    masks = sim.load_spatial_masks(fp32,gt.shape)
    blocks = sim.validate_dense16_blocks(sim.load_json_object(fp32/'spatial_split.json'),masks,gt.shape)
    indices = sim.load_index_file(run/'sample_indices.npz')
    fp_indices = sim.load_index_file(fp32/f'run_seed{meta["seed"]}'/'sample_indices.npz')
    for key in indices:
        if not np.array_equal(indices[key],fp_indices[key]):
            raise ValueError('QAT/FP32 split mismatch: '+key)
    sim.validate_sample_indices(indices,masks,gt,classes)
    batch_size = sim.resolve_qat_reference_batch_size(meta,0)
    test_blocks = [b for b in blocks if b['split']=='test']
    selection = 'manual' if args.group_index is not None else 'first_group'
    if args.prediction_dir:
        directory=Path(args.prediction_dir).expanduser().resolve()
        previous=sim.load_json_object(directory/'fpga_simulation_result.json')
        digest=hashlib.sha256((run/'best_qat_foldaware.pth').read_bytes()).hexdigest()
        if previous['qat_checkpoint_sha256']!=digest or previous['qat_reference_batch_size']!=batch_size:
            raise ValueError('Prediction directory checkpoint/batch does not match QAT run')
        one=np.load(directory/'qat_direct_prediction.npy',allow_pickle=False)
        many=np.load(directory/'qat_saved_batch_reproduced_prediction.npy',allow_pickle=False)
        saved=np.load(run/'qat_deploy_test_prediction.npy',allow_pickle=False)
        if any(a.shape!=gt.shape for a in (one,many,saved)):
            raise ValueError('Prediction shapes do not match scene')
        ix=indices['test_indices']
        if not np.array_equal(many.ravel()[ix],saved.ravel()[ix]):
            raise ValueError('Prediction directory does not reproduce saved test prediction')
        args.group_index,args.target_index=select_mismatch_group(one,many,ix,test_blocks,batch_size)
        selection='first_mismatched_test_tile_from_verified_prediction_directory'
    if args.group_index is None:
        args.group_index=0
    selected = test_blocks[args.group_index*batch_size:(args.group_index+1)*batch_size]
    if len(selected)<2 or args.target_index>=len(selected):
        raise ValueError('Select a valid group with at least two tiles and a valid target index')
    x = torch.cat([sim.make_dense16_tile(image,b)[0] for b in selected]).to(device=device,dtype=torch.float32)
    checkpoint = run/'best_qat_foldaware.pth'
    model = qat.build_configured_model(image.shape[2],classes,cfg)
    qat.prepare_qat_model(model,model_config=cfg,numeric_config=meta.get('numeric_config'),
                          ssm_contract=qat.resolve_ssm_contract(meta.get('qat_ssm_contract')))
    model.load_state_dict(sim.extract_state_dict(torch.load(checkpoint,map_location='cpu')),strict=True)
    model.to(device).eval(); qat.freeze_lsq_initialization(model)
    qat.fuse_qat_model_bns_for_deploy(model)
    before = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    init_before = [(id(m),m.init_state) for m in model.modules() if hasattr(m,'init_state')]
    out.mkdir(parents=True,exist_ok=True)
    all_rows, summaries, native = [], {}, None
    target_block = selected[args.target_index]
    h=target_block['bottom']-target_block['top'];w=target_block['right']-target_block['left']
    test_mask=np.zeros(gt.size,dtype=bool);test_mask[indices['test_indices']]=True;test_mask=test_mask.reshape(gt.shape)
    region=(slice(target_block['top'],target_block['bottom']),slice(target_block['left'],target_block['right']))
    target_mask=test_mask[region]
    expected=np.load(run/'qat_deploy_test_prediction.npy',allow_pickle=False)[region]
    # Native always runs first to anchor cross-backend comparisons.
    modes=list(dict.fromkeys(['native']+args.modes))
    for mode in modes:
        print(f'[Batch trace] {mode}, group={args.group_index}, target={args.target_index}, batch={len(selected)}',flush=True)
        with backend_mode(mode):
            flags=backend_description()
            single,sl=trace_tile(model,x[args.target_index:args.target_index+1])
            batched,bl=trace_tile(model,x,args.target_index)
            repeated,_=trace_tile(model,x,args.target_index)
            repeated_single,_=trace_tile(model,x[args.target_index:args.target_index+1])
        comparisons=[('single_vs_batch',single,batched),('batch_repeat',batched,repeated),
                     ('single_repeat',single,repeated_single)]
        if native is None:
            native=(single,batched)
        else:
            comparisons.extend([('native_single_vs_mode_single',native[0],single),
                                ('native_batch_vs_mode_batch',native[1],batched)])
        summary=dict(flags=flags,scan_backend=('torch_reference' if mode=='reference-tf32-off' or device.type=='cpu'
                  or qat._optimized_selective_scan_fn is None else 'mamba_ssm_optimized'))
        for label,a,b in comparisons:
            rows=compare_traces(a,b,mode+'/'+label,args.atol,args.rtol)
            all_rows.extend(rows);summary[label]=first_differences(rows)
        sp=sl.argmax(0).numpy()[:h,:w];bp=bl.argmax(0).numpy()[:h,:w]
        summary['predictions']=dict(valid_pixels=h*w,test_labeled_pixels=int(target_mask.sum()),
            single_vs_batch_changes_valid=int(np.count_nonzero(sp!=bp)),
            single_vs_batch_changes_test=int(np.count_nonzero((sp!=bp)&target_mask)),
            batch_vs_saved_changes_test=int(np.count_nonzero((bp!=expected)&target_mask)))
        summaries[mode]=summary
        if mode=='native' and args.save_traces:
            mapping={};vectors={}
            for i,name in enumerate(single):
                if not torch.equal(single[name][1],batched[name][1]):
                    key=f'stage_{i:04d}';mapping[key]=name
                    vectors[key+'_single']=single[name][1].numpy();vectors[key+'_batch']=batched[name][1].numpy()
            np.savez_compressed(out/'native_changed_traces.npz',**vectors)
            (out/'trace_keys.json').write_text(json.dumps(mapping,indent=2))
    if any(not torch.equal(value,model.state_dict()[k].detach().cpu()) for k,value in before.items()):
        raise RuntimeError('Diagnostic forward mutated model state')
    if init_before != [(id(m),m.init_state) for m in model.modules() if hasattr(m,'init_state')]:
        raise RuntimeError('Diagnostic forward mutated LSQ initialization')
    with (out/'batch_trace.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(all_rows[0]));writer.writeheader();writer.writerows(all_rows)
    report=dict(qat_run_dir=str(run),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        dataset=meta['dataset'],seed=meta['seed'],device=str(device),torch_version=torch.__version__,
        cuda_version=torch.version.cuda,gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else None,
        saved_batch_size=batch_size,actual_batch_size=len(selected),group_index=args.group_index,
        target_index=args.target_index,selection=selection,blocks=selected,atol=args.atol,rtol=args.rtol,model_state_unchanged=True,
        modes=summaries,scope='One target tile in its original saved test batch. No training or full-scene OA. First divergence localizes a stage, not automatically its root cause.')
    (out/'batch_diagnosis.json').write_text(json.dumps(report,indent=2))
    print('Written '+str(out/'batch_diagnosis.json'))


if __name__=='__main__':
    main()
