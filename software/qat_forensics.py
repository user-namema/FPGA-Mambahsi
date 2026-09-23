"""Opt-in validation-only diagnostics. Policies never enter training/export.

The caller passes its QAT module so this also works when training runs as __main__.
"""
import contextlib
import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch


GROUPS = ('weight', 'activation', 'dt_input', 'dt_output', 'D', 'bias')


def quantizer_group(api, name, module):
    if '.d_weight_quant.' in '.' + name:
        return 'D'
    if 'dt_proj.lsq_a' in name:
        return 'dt_input'
    if 'dt_output_quant.' in name:
        return 'dt_output'
    return 'weight' if isinstance(module, api.LsqQuantizer4weight) else 'activation'


@contextlib.contextmanager
def preserve_execution(model=None, loaders=()):
    """Additional evaluation must not consume training/global RNG state."""
    py, np_state, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    generators = [(l.generator, l.generator.get_state()) for l in loaders
                  if getattr(l, 'generator', None) is not None]
    modes = [(m, m.training) for m in model.modules()] if model is not None else []
    try:
        yield
    finally:
        random.setstate(py)
        np.random.set_state(np_state)
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)
        for generator, state in generators:
            generator.set_state(state)
        for module, mode in modes:
            module.training = mode


@contextlib.contextmanager
def quantization_policy(api, model, enabled):
    """Bypass LSQ without calibration/state changes; separately bypass all biases.

Scales are the SAME calibrated values in every case. A bypass returns the real
tensor and its scale metadata, not a newly initialized 32-bit quantizer.
"""
    enabled = set(enabled)
    if enabled - set(GROUPS):
        raise ValueError('Unknown quantization group')
    restores = []
    original_bias = api._quantize_accumulator_bias
    try:
        for name, module in model.named_modules():
            if not isinstance(module, (api.LsqQuantizer4input, api.LsqQuantizer4weight)):
                continue
            if quantizer_group(api, name, module) not in enabled:
                restores.append((module, 'forward' in module.__dict__, module.__dict__.get('forward')))
                scale = module.s[0] if isinstance(module, api.LsqQuantizer4input) else module.s
                module.forward = lambda value, scale=scale: (value, scale)
        if 'bias' not in enabled:
            api._quantize_accumulator_bias = lambda bias, scale_x, weight_scale: bias
        yield
    finally:
        api._quantize_accumulator_bias = original_bias
        for module, existed, value in restores:
            if existed:
                module.forward = value
            else:
                del module.forward


def runtime_metadata(api):
    paths = [Path(api.__file__), Path(__file__)]
    return dict(torch_version=torch.__version__, cuda_version=torch.version.cuda,
                selective_scan_backend=api.selective_scan_backend(),
                matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_tf32=torch.backends.cudnn.allow_tf32,
                cudnn_deterministic=torch.backends.cudnn.deterministic,
                cudnn_benchmark=torch.backends.cudnn.benchmark,
                source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def require_calibration_frozen(api, model):
    for name, module in model.named_modules():
        if isinstance(module, (api.LsqQuantizer4input, api.LsqQuantizer4weight)):
            if module.init_state < module.batch_init:
                raise RuntimeError('Freeze LSQ calibration before diagnosis: '+name)


@contextlib.contextmanager
def weight_initialization_policy(api, model, policy):
    """Switch only ordinary weight scales; use actual BN-folded weights."""
    if policy not in ('mean', 'patch_max', 'all_weight_max'):
        raise ValueError('Unknown weight initialization policy: '+policy)
    require_calibration_frozen(api, model)
    saved, changes = [], []
    try:
        for name, layer in model.named_modules():
            if not isinstance(layer, (api.QuanConv, api.QuanConv1d, api.QuanLinear)):
                continue
            if policy == 'mean' or (policy == 'patch_max' and name != 'patch_embedding.0'):
                continue
            weight = layer.weight.detach()
            if getattr(layer, 'norm', False):
                factor = layer.bn_weight.detach()/torch.sqrt(layer.bn_running_var+layer.bn_eps)
                weight = weight*factor.reshape([-1]+[1]*(weight.ndim-1))
            scale = layer.lsq_w.s
            old = scale.detach().clone()
            saved.append((scale, old))
            new = (weight.abs().max()/layer.lsq_w.Qp).clamp(min=1e-8)
            with torch.no_grad():
                scale.copy_(new.reshape_as(scale))
            changes.append(dict(layer=name, old_scale=float(old.item()), new_scale=float(scale.item()),
                                folded_bn=bool(getattr(layer,'norm',False))))
        if policy == 'patch_max' and len(changes) != 1:
            raise ValueError('Expected exactly one patch_embedding.0 quantized layer')
        yield changes
    finally:
        with torch.no_grad():
            for scale, value in saved:
                scale.copy_(value)


@torch.no_grad()
def apply_weight_initialization(api, model, policy):
    """Commit the same scale policy used by the validation-only comparison."""
    with weight_initialization_policy(api, model, policy) as changes:
        changes = [dict(row) for row in changes]
    named = dict(model.named_modules())
    for row in changes:
        named[row['layer']].lsq_w.s.fill_(row['new_scale'])
    return changes


@torch.no_grad()
def weight_init_validation(api, model, args, loader, labels, classes, device, out, logger, fp32_metrics):
    """Three full-validation cases from one normal calibration, with state audit."""
    require_calibration_frozen(api, model)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    before = cpu_state(model)
    report = dict(scope='complete saved validation split; no optimizer or test evaluation',
                  runtime=runtime_metadata(api), run=out.parent.name, model_config=args.model_config,
                  eval_batch_size=args.eval_batch_size, calibration_steps=args.calibration_steps,
                  fp32_validation=fp32_metrics,
                  scale_protocol='One normal mean-weight calibration. Activation/dt/D scales stay fixed. '
                    'Max uses folded ordinary weights / Qp. Existing bias quantization uses the new accumulator scale.',
                  cases={})
    baseline = None
    with preserve_execution(model, [loader]):
        model.eval()
        for policy in ('mean','patch_max','all_weight_max'):
            with weight_initialization_policy(api, model, policy) as changes:
                allowed = {id(layer.lsq_w.s) for name,layer in model.named_modules()
                           if any(c['layer']==name for c in changes)}
                # state_dict contains aliases; permit only the selected scale parameters.
                allowed_keys = {k for k,v in model.state_dict(keep_vars=True).items() if id(v) in allowed}
                if any(not torch.equal(v,model.state_dict()[k].cpu())
                       for k,v in before.items() if k not in allowed_keys):
                    raise RuntimeError('Weight policy changed a non-target tensor')
                pred, visits, metrics = api.evaluate_model(model, loader, labels.shape,
                    args.tile_size, device, labels, classes)
                if not np.all(visits[labels>=0] == 1):
                    raise AssertionError('Validation coverage mismatch')
                if baseline is None:
                    baseline = pred.copy()
                report['cases'][policy] = dict(validation=metrics, changed_scales=changes,
                    OA_minus_mean_pp=100*(metrics['OA']-report['cases'].get('mean',{'validation':metrics})['validation']['OA']),
                    prediction_match_mean=float((pred[labels>=0]==baseline[labels>=0]).mean()))
                # Store only validation labels/predictions; no full-scene background/test output.
                np.savez_compressed(out/(policy+'_validation_prediction.npz'),
                    indices=np.flatnonzero(labels>=0), target=labels[labels>=0], prediction=pred[labels>=0])
            if any(not torch.equal(v, model.state_dict()[k].cpu()) for k,v in before.items()):
                raise RuntimeError('Weight initialization diagnosis did not restore model state')
            logger.info('Weight init=%s val_OA=%.6f val_mAcc=%.6f changed_scales=%d',
                        policy, metrics['OA'], metrics['mAcc'], len(changes))
            api.save_json(out/'weight_init_validation.json', report)
    report['model_state_unchanged'] = True
    api.save_json(out/'weight_init_validation.json', report)
    with (out/'weight_init_summary.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['case','val_OA','val_mAcc','val_Kappa','OA_minus_mean_pp','changed_weight_scales'])
        writer.writeheader()
        for name,row in report['cases'].items():
            v=row['validation']
            writer.writerow(dict(case=name,val_OA=v['OA'],val_mAcc=v['mAcc'],val_Kappa=v['Kappa'],
                OA_minus_mean_pp=row['OA_minus_mean_pp'],changed_weight_scales=len(row['changed_scales'])))
    return report


def tensor_difference(a, b, atol=1e-6, rtol=1e-5):
    if a.shape != b.shape:
        raise ValueError('Trace shapes differ')
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise FloatingPointError('Nonfinite diagnostic tensor')
    a, b = a.double(), b.double()
    delta = (a-b).abs()
    return dict(count=a.numel(), changed=int((a != b).sum()),
                changed_fraction=float((a != b).double().mean()),
                above_tolerance=int((delta > atol + rtol*a.abs()).sum()),
                mae=float(delta.mean()), max_abs_error=float(delta.max()))


@torch.no_grad()
def initial_quantization_diagnosis(api, model, fp32_state, args, val_loader,
                                  val_label, classes, in_channels, device, out, logger):
    """Full saved validation split; no test evaluation or per-case recalibration."""
    out = Path(out)
    require_calibration_frozen(api, model)
    out.mkdir(parents=True, exist_ok=False)
    before = cpu_state(model)
    report = dict(scope='saved validation labels only', runtime=runtime_metadata(api),
                  run=out.parent.name, model_config=args.model_config,
                  calibration_steps=args.calibration_steps,
                  eval_batch_size=args.eval_batch_size, atol=1e-6, rtol=1e-5,
                  interpretation='Fixed calibrated scales. Only-group and leave-one-out effects are not additive.',
                  quantizer_groups={name: quantizer_group(api, name, m)
                    for name, m in model.named_modules()
                    if isinstance(m, (api.LsqQuantizer4input, api.LsqQuantizer4weight))}, cases={})
    def evaluate(m):
        # Capture full validation output tensors, not only argmax/confusion counts.
        outputs = []
        handle = m.register_forward_hook(lambda module, inputs, result: outputs.append(result.detach().cpu().clone()))
        try:
            pred, visits, metrics = api.evaluate_model(m, val_loader, val_label.shape,
                args.tile_size, device, val_label, classes)
        finally:
            handle.remove()
        if not np.all(visits[val_label >= 0] == 1):
            raise AssertionError('Validation coverage mismatch')
        return pred, metrics, torch.cat([x.flatten() for x in outputs])
    with preserve_execution(model, [val_loader]):
        fp32 = api.build_fp32_model(in_channels, classes, fp32_state, device, args.model_config)
        fp_pred, fp_metrics, fp_logits = evaluate(fp32)
        del fp32
        report['fp32'] = fp_metrics
        policies = [('none', ()), ('all', GROUPS)]
        policies += [('only_' + g, (g,)) for g in GROUPS]
        policies += [('without_' + g, tuple(x for x in GROUPS if x != g)) for g in GROUPS]
        policies += [('all_repeat', GROUPS)]
        all_pred, all_logits = None, None
        for name, enabled in policies:
            with quantization_policy(api, model, enabled):
                pred, metrics, logits = evaluate(model)
            row = dict(enabled_groups=list(enabled), validation=metrics,
                       prediction_match_fp32=float((pred[val_label >= 0] == fp_pred[val_label >= 0]).mean()),
                       output_vs_fp32=tensor_difference(fp_logits, logits))
            if name == 'all':
                all_pred, all_logits = pred.copy(), logits.clone()
            if all_pred is not None:
                row['prediction_match_all'] = float((pred[val_label >= 0] == all_pred[val_label >= 0]).mean())
                row['OA_minus_all_pp'] = 100*(metrics['OA'] - report['cases'].get('all', row)['validation']['OA'])
            if name == 'all_repeat':
                row['output_vs_all'] = tensor_difference(all_logits, logits)
            report['cases'][name] = row
            logger.info('Initial quantization case=%s val_OA=%.6f val_mAcc=%.6f FP32_match=%.6f',
                        name, metrics['OA'], metrics['mAcc'], row['prediction_match_fp32'])
            api.save_json(out/'initial_quantization.json', report)
    report['model_state_unchanged'] = all(torch.equal(v, model.state_dict()[k].cpu()) for k, v in before.items())
    report['conversion_within_tolerance'] = report['cases']['none']['output_vs_fp32']['above_tolerance'] == 0
    report['conversion_predictions_match'] = report['cases']['none']['prediction_match_fp32'] == 1.0
    api.save_json(out/'initial_quantization.json', report)
    with (out/'initial_quantization_summary.csv').open('w', newline='') as stream:
        fields = ['case','val_OA','val_mAcc','FP32_match','output_MAE_vs_FP32','output_MaxAE_vs_FP32']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow(dict(case='fp32', val_OA=report['fp32']['OA'], val_mAcc=report['fp32']['mAcc'],
                             FP32_match=1., output_MAE_vs_FP32=0., output_MaxAE_vs_FP32=0.))
        for name, row in report['cases'].items():
            writer.writerow(dict(case=name, val_OA=row['validation']['OA'], val_mAcc=row['validation']['mAcc'],
                FP32_match=row['prediction_match_fp32'], output_MAE_vs_FP32=row['output_vs_fp32']['mae'],
                output_MaxAE_vs_FP32=row['output_vs_fp32']['max_abs_error']))
    if not report['model_state_unchanged']:
        raise RuntimeError('Initial diagnosis mutated model state')
    return report


def select_regression_batch(dataset, previous, current, labels, batch_size):
    regression = (labels >= 0) & (previous == labels) & (current != labels)
    scores = []
    for start in range(0, len(dataset), batch_size):
        score = sum(int(regression[t:b, l:r].sum()) for t,b,l,r,_ in dataset.tiles[start:start+batch_size])
        scores.append(score)
    start = int(np.argmax(scores))*batch_size
    items = [dataset[i] for i in range(start, min(start+batch_size, len(dataset)))]
    def batch_prediction(scene):
        rows = []
        for item in items:
            top, left, height, width = item[2:6]
            tile = np.full(tuple(item[1].shape), -1, dtype=np.int64)
            tile[:height,:width] = scene[top:top+height,left:left+width]
            rows.append(torch.from_numpy(tile))
        return torch.stack(rows)
    return dict(inputs=torch.stack([x[0] for x in items]), labels=torch.stack([x[1] for x in items]),
                expected_before_prediction=batch_prediction(previous),
                expected_after_prediction=batch_prediction(current),
                dataset_indices=list(range(start, start+len(items))),
                tiles=[list(x[2:]) for x in items], regressed_pixels=max(scores))


class JumpCapture:
    """Keep one prior evaluated state in RAM; persist only bounded OA-drop pairs.

    These are inference replay checkpoints, not optimizer/resume checkpoints.
    """
    def __init__(self, api, run_dir, args, classes, in_channels, numeric, contract):
        self.api, self.args = api, args
        self.root = Path(run_dir)/'jump_events'
        self.previous = None
        self.events = []
        self.metadata = dict(model_config=args.model_config, classes=classes, in_channels=in_channels,
                             run_dir=str(Path(run_dir).resolve()),
                             numeric_config=numeric.to_dict(), ssm_contract=contract,
                             nbit=args.nbit, tile_size=args.tile_size, runtime=runtime_metadata(api),
                             eval_batch_size=args.eval_batch_size,
                             training_device=getattr(args, 'device', 'unspecified'),
                             checkpoint_kind='inference-only fold-aware state_dict')

    def observe(self, model, epoch, metrics, prediction, dataset, labels):
        state = cpu_state(model)
        if self.previous is not None:
            prev_epoch, prev_metrics, prev_prediction, prev_state = self.previous
            drop = 100*(prev_metrics['OA'] - metrics['OA'])
            if (epoch >= self.args.capture_start_epoch and drop >= self.args.capture_oa_drop_pp
                    and len(self.events) < self.args.capture_max_events):
                event = self.root/f'epoch{prev_epoch:04d}_to_{epoch:04d}'
                event.mkdir(parents=True, exist_ok=False)
                batch = select_regression_batch(dataset, prev_prediction, prediction, labels, self.args.eval_batch_size)
                torch.save(prev_state, event/'before.pth')
                torch.save(state, event/'after.pth')
                torch.save(batch, event/'validation_batch.pt')
                meta = dict(self.metadata, before_epoch=prev_epoch, after_epoch=epoch, OA_drop_pp=drop,
                            adjacent_epochs=(epoch == prev_epoch+1), before_validation=prev_metrics,
                            after_validation=metrics, selection='saved-order batch with most correct-to-wrong validation pixels',
                            dataset_indices=batch['dataset_indices'], regressed_pixels=batch['regressed_pixels'])
                self.api.save_json(event/'event.json', meta)
                self.events.append(str(event))
                self.api.save_json(self.root/'manifest.json', dict(events=self.events,
                    max_events=self.args.capture_max_events, threshold_pp=self.args.capture_oa_drop_pp))
                print('[QAT jump captured] '+str(event), flush=True)
        self.previous = (epoch, metrics, prediction.copy(), state)


@torch.no_grad()
def trace_quantized_batch(api, model, inputs, tile_size):
    """Execution-ordered whole-batch LSQ, layer and SSM outputs, including folded weights."""
    require_calibration_frozen(api, model)
    traces, stats, handles, counts = {}, {}, [], {}
    original_bias = api._quantize_accumulator_bias
    def save(name, tensor, kind):
        index = counts.get(name, 0)
        counts[name] = index+1
        key = name+'#'+str(index)
        traces[key] = (kind, tensor.detach().cpu().clone())
        return key
    def collect(name):
        def hook(module, args, result):
            if isinstance(module, (api.LsqQuantizer4input, api.LsqQuantizer4weight)):
                raw = args[0]/module.s
                codes = api._LsqRound.apply(raw.clamp(module.Qn, module.Qp)).to(torch.int32)
                key = save(name+'/codes', codes, 'codes')
                save(name+'/output', result[0], 'float')
                # Distance to the nearest ACTIVE half-integer decision boundary.
                lo = torch.floor(raw).clamp(module.Qn, module.Qp-1)
                distance = (raw-(lo+.5)).abs()
                stats[key] = dict(scale=float(module.s.item()),
                    group=quantizer_group(api, name, module),
                    threshold_distance_min=float(distance.min()),
                    threshold_distance_mean=float(distance.mean()),
                    near_threshold_001_fraction=float((distance < .01).float().mean()),
                    clipping_fraction=float(((raw < module.Qn)|(raw > module.Qp)).float().mean()))
            elif isinstance(result, torch.Tensor):
                save(name+'/output', result, 'float')
        return hook
    def trace_bias(bias, scale_x, weight_scale):
        output = original_bias(bias, scale_x, weight_scale)
        if bias is not None and scale_x is not None:
            scale = (scale_x.reshape(-1)[0]*weight_scale.reshape(-1)[0]).detach()
            codes = torch.round(bias/scale).clamp(api.BIAS_ACCUMULATOR_MIN, api.BIAS_ACCUMULATOR_MAX).to(torch.int64)
            key = save('accumulator_bias/codes', codes, 'codes')
            stats[key] = dict(scale=float(scale), group='bias')
            save('accumulator_bias/output', output, 'float')
        return output
    types = (api.LsqQuantizer4input, api.LsqQuantizer4weight, api.QuanConv,
             api.QuanConv1d, api.QuanLinear, api.CurrentMambaCore, api.BothMamba)
    with preserve_execution(model):
        model.eval()
        try:
            api._quantize_accumulator_bias = trace_bias
            for name, module in model.named_modules():
                if isinstance(module, types):
                    handles.append(module.register_forward_hook(collect(name)))
            logits = api.dense_logits(model, inputs, tile_size).detach().cpu()
            save('dense_logits', logits, 'float')
        finally:
            api._quantize_accumulator_bias = original_bias
            for handle in handles:
                handle.remove()
    return traces, stats, logits


def compare_trace(reference, actual):
    if list(reference) != list(actual):
        raise ValueError('Execution order differs between traces')
    rows = []
    for order, (name, (kind, value)) in enumerate(reference.items()):
        other_kind, other = actual[name]
        if kind != other_kind:
            raise ValueError('Trace kinds differ')
        rows.append(dict(order=order, trace=name, kind=kind, **tensor_difference(value, other)))
    return rows
