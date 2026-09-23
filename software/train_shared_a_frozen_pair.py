"""Train shared/per-channel A on saved current-architecture splits and exact indices.
Reuses the existing FP32 training loop; never refits PCA or resamples labels.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--a-mode', choices=['shared', 'per_channel'], required=True)
    p.add_argument('--seeds', default='0,1,2,3,4,5,6,7,8,9')
    p.add_argument('--data-path', default='./data')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--max-epoch', type=int, default=400)
    p.add_argument('--resume', action='store_true')
    opt = p.parse_args()
    import numpy as np
    import torch
    import train_mambahsi_spatial_split_128_dense as train
    from train_mambahsi_spatial_split_dense_qat import load_dataset
    from analyze_alog_dynamics import apply_saved_preprocess, MODEL_ARGUMENTS

    source = Path(opt.source_dir).resolve()
    out = Path(opt.output_dir).resolve()
    seeds = [int(x) for x in opt.seeds.split(',')]
    if len(set(seeds)) != len(seeds) or min(seeds) < 0 or opt.max_epoch < 1:
        raise ValueError('Invalid seeds/max-epoch')
    split = json.loads((source / 'spatial_split.json').read_text())
    dataset = split['dataset']
    if int(split['effective_tile_size']) != 16 or split['strategy'] != 'blocks':
        raise ValueError('Requires saved dense16 blocks protocol')
    c = json.loads((source / ('run_seed%d' % seeds[0]) / 'model_config.json').read_text())
    expected = dict(hidden_dim=32, branch_mode='both', fusion_mode='sum', skip_scale=2,
                    use_D=True, use_z=False, A_mode='shared', norm_path='bn', activation='relu',
                    head_dim=64, token_num=4, d_state=16)
    if any(c.get(k) != v for k, v in expected.items()):
        raise ValueError('Source is not the current D=1,z=0 shared-A architecture: %s' % c)
    device = torch.device(opt.device)
    if device.type != 'cuda' or not torch.cuda.is_available() or train.selective_scan_backend() != 'mamba_ssm_optimized':
        raise RuntimeError('Paired training requires CUDA and optimized selective scan')
    torch.cuda.set_device(device)
    records = {}
    for seed in seeds:
        run = source / ('run_seed%d' % seed)
        meta = json.loads((run / 'result.json').read_text())
        config = json.loads((run / 'model_config.json').read_text())
        if any(config.get(k) != v for k, v in expected.items()) or meta['dataset'] != dataset or int(meta['seed']) != seed:
            raise ValueError('Source metadata mismatch: %s' % run)
        records[seed] = (meta, run / 'sample_indices.npz')
        if not records[seed][1].is_file(): raise FileNotFoundError(records[seed][1])
    signature = dict(dataset=dataset, source=str(source), A_mode=opt.a_mode, seeds=seeds,
                     max_epoch=opt.max_epoch,
                     source_hashes={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in [source/'spatial_split.json', source/'train_only_preprocess.npz'] + [v[1] for v in records.values()] + [source/('run_seed%d'%seed)/name for seed in seeds for name in ['model_config.json','result.json']]})
    manifest = out / 'pair_protocol.json'
    if out.exists() and any(out.iterdir()):
        if not opt.resume or not manifest.exists() or json.loads(manifest.read_text()) != signature:
            raise FileExistsError('Output exists or protocol differs; choose a new output directory')
    out.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(signature, indent=2))
    for name in ['spatial_split.json', 'spatial_split_masks.npz', 'train_only_preprocess.npz']:
        shutil.copy2(source/name, out/name)
    raw, gt, classes = load_dataset(dataset, opt.data_path)
    raw, gt, classes = train.validate_data(raw, gt, dataset, 16)
    image = apply_saved_preprocess(raw, source/'train_only_preprocess.npz')
    regions = {}
    for name in ['train', 'val', 'test']:
        key = 'validation' if name == 'val' else name
        regions[name] = [(b['top'], b['bottom'], b['left'], b['right']) for b in split['blocks'] if b['split'] == key]
        split[name+'_regions'] = regions[name]
    masks = [train.regions_to_mask(gt.shape, regions[n]) for n in ['train','val','test']]
    if not np.all(sum(m.astype(np.int8) for m in masks) == 1):
        raise ValueError('Spatial regions must partition the scene exactly')
    original_labels = train.create_label_maps
    try:
        for seed in seeds:
            target = out / ('run_seed%d' % seed)
            if target.exists():
                if opt.resume and (target/'result.json').exists() and (target/'best_model.pth').exists(): continue
                raise FileExistsError('Incomplete run preserved: %s; use a new output directory' % target)
            meta, index_path = records[seed]
            indices = np.load(index_path)
            frozen = []
            for name, mask in zip(['train','val','test'], masks):
                idx = np.asarray(indices[name+'_indices'], dtype=np.int64).reshape(-1)
                if len(np.unique(idx)) != len(idx) or np.any(idx < 0) or np.any(idx >= gt.size):
                    raise ValueError('Invalid or duplicate sample indices')
                if not np.all(mask.ravel()[idx]) or np.any(gt.ravel()[idx] <= 0):
                    raise ValueError('Saved indices violate spatial/label contract')
                frozen.append(idx)
            def frozen_labels(*unused):
                maps = []
                for idx in frozen:
                    lab = np.full(gt.size, -1, dtype=np.int64)
                    lab[idx] = gt.ravel()[idx] - 1
                    maps.append(lab.reshape(gt.shape))
                return tuple(maps + [x.copy() for x in frozen])
            train.create_label_maps = frozen_labels
            args = train.build_parser().parse_args([])
            for field in MODEL_ARGUMENTS: setattr(args, field, c[field])
            args.A_mode = opt.a_mode
            args.model_variant = 'current'
            train.resolve_model_arguments(args)
            args.tile_size = args.split_block_size = 16
            args.seeds = seeds
            args.max_epoch = opt.max_epoch
            args.eval_batch_size = 8
            args.eval_interval = 5
            args.scheduler_step_size, args.scheduler_gamma = 20, 0.9
            args.lr = float(meta['initial_lr'])
            args.weight_decay = float(meta['weight_decay'])
            args.batch_size = int(meta['batch_size'])
            args.train_samples = int(meta['train_samples_per_class'])
            args.class_weight = meta['class_weight_mode']
            args.grad_clip_norm = float(meta['grad_clip_norm'])
            args.scheduler = meta['scheduler']
            args.selection_metric = meta['selection_metric']
            args.early_stopping_patience = int(meta['early_stopping_patience'])
            args.early_stopping_min_delta = float(meta['early_stopping_min_delta'])
            logger = train.make_logger(out/('train_seed%d.log' % seed), '%s_%s_%d' % (dataset,opt.a_mode,seed))
            # Both variants train from scratch with identical protocol. Only A_mode differs.
            train.run_seed(args,dataset,image,gt,classes,split,*masks,out,seed,device,logger)
            written = np.load(target/'sample_indices.npz')
            assert all(np.array_equal(written[n+'_indices'], idx) for n,idx in zip(['train','val','test'],frozen))
            (target/'training_arguments.json').write_text(json.dumps(vars(args), indent=2, default=str))
    finally:
        train.create_label_maps = original_labels


if __name__ == '__main__': main()
