#!/usr/bin/env python3
"""One-checkpoint D1 N0/N1/N2/N3 paired experiment on the current eval1 graph.

Frozen historical approximation algorithms are reused, not the UP-only wrapper
or its checkpoint-specific anchors. Every checkpoint gets its own constants.
"""
import argparse
import contextlib
import inspect
import io
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
import both_FPGA_single_qat_source as base
from ssm_error_ablation import SSMNumericConfig
from backend_error_metrics import numeric, paired_labels, scene_from_tiles

REFERENCE = Path(__file__).parent / 'snapshots/nonlinear_D1_20260916'
sys.path.append(str(REFERENCE))
from d1_methods import CORES, FIT, constants, variants, export_tables, digest, dump, stats, merge, finished

VERSION = 'D1-four-datasets-20260927-v1'


class Recorder:
    def __init__(self, experiment, category, method, downstream=None):
        self.ex, self.category, self.method, self.downstream = experiment, category, method, downstream

    def add(self, core, time, actual, reference, saturation=0):
        ex = self.ex
        if self.downstream is not None:
            self.downstream.add(core, time, actual, reference, saturation)
        # Lightweight full-scene saturation coverage for every backend.
        if self.category == 'end_to_end' and core.endswith('/state'):
            row = ex.saturations.setdefault(self.method, {}).setdefault(core, dict(count=0, saturations=0))
            row['count'] += actual.numel()
            row['saturations'] += int(saturation)
        if not ex.traced():
            return
        key = (core, int(time))
        if self.method == 'n3':
            ex.states[key] = actual.detach().cpu().clone()
        else:
            row = ex.trace.setdefault(self.category, {}).setdefault(self.method, {}).setdefault(f'{core}@{time}', {})
            merge(row, stats(ex.states[key], actual))
            row['saturations'] = row.get('saturations', 0) + int(saturation)
        row = ex.float_errors.setdefault(self.category, {}).setdefault(self.method, {}).setdefault(f'{core}@{time}', {})
        merge(row, stats(reference, actual))


class Experiment:
    def __init__(self, trace_tiles=5):
        self.trace_tiles = trace_tiles
        self.method, self.tile = 'n3', -1
        self.methods = ('n3', 'n0', 'n1', 'n2')
        self.logits = {m: [] for m in self.methods}
        self.float_predictions = {m: [] for m in self.methods}
        self.tables, self.parameters, self.table_names = {}, {}, {}
        self.trace, self.float_errors, self.layers, self.layer_float = {}, {}, {}, {}
        self.states, self.boundaries, self.saturations = {}, {}, {}
        self.placements = None
        self.scale = None
        self.original = {}

    def traced(self):
        return self.trace_tiles < 0 or self.tile < self.trace_tiles

    def coefficients(self, m, sdt, sb, su, name, output_dir, device):
        result = self.original['get_or_build_ssm_luts'](m, sdt, sb, su, name,
                         output_dir if self.method == 'n3' else None, device)
        if not getattr(m, 'use_D', False):
            raise ValueError(f'{name}: this experiment requires a D1 checkpoint')
        cache = m.__dict__['_fpga_ssm_lut_cache']
        signature = cache['content_signature_sha256']
        if name in self.parameters and self.parameters[name]['signature'] != signature:
            raise ValueError(f'{name}: coefficient contract changed during evaluation')
        if name not in self.tables:
            p = constants(m.A_log_shared, cache['s_dt'], cache['s_b'], cache['s_u'], result[3])
            p['signature'] = signature
            p['dt_physical_domain'] = [-128*p['s_dt'], 127*p['s_dt']]
            # Never extrapolate N2 or refit using the evaluated scene.
            if max(abs(v) for v in p['dt_physical_domain']) > 16:
                raise ValueError(f'{name}: N2 frozen fit [-16,16] does not cover {p["dt_physical_domain"]}; '
                                 'run invalid, not silently clipped or automatically refitted')
            self.parameters[name] = p
            self.tables[name] = variants(cache['tables'], p, device)
            p['coefficient_code_limits'] = {
                method: dict(K_at_unsigned_max=int((table['k'] == (1<<19)-1).sum()),
                             A_at_one=int((table['a'] == 1<<24).sum()),
                             A_at_zero=int((table['a'] == 0).sum()))
                for method, table in self.tables[name].items()}
        self.table_names[id(cache['tables'])] = name
        return result

    def scan(self, u, dt, b, readout, tables, config=None, scales=(1., 1.),
             recorder=None, core='', d_tables=None, components=None):
        if d_tables is None or self.table_names.get(id(tables)) != core:
            raise ValueError(f'{core}: D table/cache contract mismatch')
        if 'd_coefficient' not in self.parameters[core]:
            self.parameters[core] = export_tables(self.out/'coefficients', core,
                          self.tables[core], self.parameters[core], d_tables)
        p = self.parameters[core]
        if p['d_coefficient'] != d_tables['coefficient'].cpu().tolist() or p['d_left_shift'] != d_tables['left_shift']:
            raise ValueError(f'{core}: D constants changed across methods')
        actual = self.original['scan_codes'](u, dt, b, readout, self.tables[core][self.method], config,
            scales=scales, recorder=Recorder(self, 'end_to_end', self.method,
                                           recorder if self.method == 'n3' else None),
            core=core, d_tables=d_tables, components=components if self.method == 'n3' else None)
        if self.method == 'n3' and self.traced():
            for method in self.methods[1:]:
                self.original['scan_codes'](u, dt, b, readout, self.tables[core][method], config,
                    scales=scales, recorder=Recorder(self, 'isolated', method), core=core, d_tables=d_tables)
        return actual

    def tile_run(self, net, tile, scale, output_dir, device):
        self.tile += 1
        self.states.clear()
        self.boundaries.clear()
        flags = [(m, m._capture_ssm_inputs) for m in net.modules() if hasattr(m, '_capture_ssm_inputs')]
        result = None
        try:
            for method in self.methods:
                self.method = method
                if method != 'n3':
                    for m, _ in flags:
                        m._capture_ssm_inputs = False
                with contextlib.redirect_stdout(io.StringIO()) if method != 'n3' else contextlib.nullcontext():
                    value = self.original['simulate_tile_dual_int8'](net, tile, scale,
                                       output_dir if method == 'n3' else None, device)
                if result is None:
                    result = value
                elif not torch.equal(result[0], value[0]) or not torch.equal(result[1], value[1]):
                    raise AssertionError('Floating QAT/staged reference changed between methods')
                q = value[2].detach().cpu().numpy()
                if not np.all(q == np.round(q)) or np.any((q < -128) | (q > 127)):
                    raise ValueError('Invalid head INT8 output')
                s = float(value[3])
                if self.scale is None:
                    self.scale = s
                if s != self.scale:
                    raise ValueError('Head scale changed across methods or tiles')
                self.logits[method].append(q[0].astype(np.int8))
                # Same floating postprocess as current eval1; recorded separately.
                up = F.interpolate(value[2].float()*base.scalar_scale(value[3], device),
                                   size=(16, 16), mode='bilinear', align_corners=True)
                self.float_predictions[method].append(up.argmax(dim=1)[0].cpu().numpy().astype(np.uint8))
        finally:
            self.method = 'n3'
            for m, flag in flags:
                m._capture_ssm_inputs = flag
        return result

    def report(self, *args, **kwargs):
        result = self.original['report_fixed_float'](*args, **kwargs)
        if self.tile < 0:
            return result
        v = inspect.signature(self.original['report_fixed_float']).bind(*args, **kwargs).arguments
        value = v.get('codes') if v.get('codes') is not None else v.get('fixed')
        if value is None:
            return result
        name = v['name']
        if self.method == 'n3':
            self.boundaries[name] = torch.as_tensor(value).detach().cpu().clone()
        else:
            merge(self.layers.setdefault(self.method, {}).setdefault(name, {}), stats(self.boundaries[name], value))
        physical = v.get('fixed')
        if physical is None:
            physical = torch.as_tensor(value).double()*torch.as_tensor(v['scale'], device=value.device).double()
        merge(self.layer_float.setdefault(self.method, {}).setdefault(name, {}), stats(v['reference'], physical))
        return result

    def capture_inputs(self, image, blocks, scale, output_dir, device, **kwargs):
        self.placements = [dict(block, valid_height=min(16, image.shape[0]-block['top']),
                                valid_width=min(16, image.shape[1]-block['left'])) for block in blocks]
        return self.original['export_pcie_input_tiles'](image, blocks, scale, output_dir, device, **kwargs)

    def evaluate(self, *args, **kwargs):
        v = inspect.signature(self.original['evaluate_predictions']).bind(*args, **kwargs).arguments
        self.gt, self.indices = np.asarray(v['gt']), v['sample_indices']
        self.qat_pred = np.asarray(v['predict_true']).copy()
        self.base_pred = np.asarray(v['predict_int8']).copy()
        q = v['final_true'].detach().cpu().numpy()[0]
        top = np.partition(q, -2, axis=0)[-2:]
        self.qat_margin = top[-1]-top[-2]
        return self.original['evaluate_predictions'](*args, **kwargs)

    def install(self):
        mapping = dict(get_or_build_ssm_luts=self.coefficients, scan_codes=self.scan,
                       simulate_tile_dual_int8=self.tile_run, report_fixed_float=self.report,
                       export_pcie_input_tiles=self.capture_inputs, evaluate_predictions=self.evaluate)
        for name, callback in mapping.items():
            self.original[name] = getattr(base, name)
            setattr(base, name, callback)

    def restore(self):
        for name, callback in self.original.items():
            setattr(base, name, callback)

    def finish(self, result):
        if set(self.tables) != set(CORES) or self.placements is None:
            raise AssertionError('Incomplete six-core experiment')
        arrays = {m: np.stack(v) for m, v in self.logits.items()}
        if any(len(v) != len(self.placements) for v in arrays.values()):
            raise AssertionError('Incomplete scene')
        predictions, margins, floats = {}, {}, {}
        for m, a in arrays.items():
            predictions[m], margins[m] = scene_from_tiles(a, self.placements, self.gt.shape, self.scale)
            floats[m] = np.zeros(self.gt.shape, dtype=np.uint8)
            for tile, p in zip(self.float_predictions[m], self.placements):
                y, x, h, w = (p[k] for k in ('top', 'left', 'valid_height', 'valid_width'))
                floats[m][y:y+h, x:x+w] = tile[:h, :w]
            np.save(self.out/f'{m}_tile_logits_int8.npy', a)
            np.save(self.out/f'{m}_prediction_integer.npy', predictions[m])
            np.save(self.out/f'{m}_prediction_float_interpolation.npy', floats[m])
        if not np.array_equal(floats['n3'], self.base_pred):
            raise AssertionError('N3 does not reproduce unmodified base postprocess')
        np.save(self.out/'ground_truth.npy', self.gt)
        np.save(self.out/'n3_margin_physical.npy', margins['n3'])
        np.save(self.out/'qat_margin_physical.npy', self.qat_margin)
        np.save(self.out/'qat_direct_prediction.npy', self.qat_pred)
        np.savez(self.out/'sample_indices.npz', **self.indices)
        scopes = dict(scene=np.arange(self.gt.size), labeled=np.flatnonzero(self.gt.reshape(-1)>0),
                      train=self.indices['train_indices'], validation=self.indices['val_indices'], test=self.indices['test_indices'])
        summary = dict(schema=VERSION, status='COMPLETE', dataset=result['dataset'], seed=result['seed'],
                       checkpoint_sha256=result['qat_checkpoint_sha256'], numeric_config=result['numeric_config'],
                       tiles=len(self.placements), class_count=result['class_count'], head_scale=self.scale,
                       n2_fit=FIT, coefficients=self.parameters, methods={},
                       trace_tiles=min(len(self.placements), self.trace_tiles) if self.trace_tiles>=0 else len(self.placements),
                       full_scene_state_saturations=self.saturations,
                       hardware_validation_claim=False, n3_float_postprocess_matches_base=True,
                       scope='Frozen per-run D1 graph; only integer Abar/K backend changes; common D and readout',
                       postprocess='Exact score numerators /25; first argmax; interpolate 16x16 then crop',
                       float_interpolation_scope='Floating align_corners audit only; margin reference remains the exact N3 score margin',
                       saturation_scope='Exact full-scene state clipping counts; coefficient endpoint counts are NOT clipping counts; outer-layer clipping not instrumented',
                       source_sha256={str(p.relative_to(Path(__file__).parent)): digest(p) for p in
                                      [Path(__file__), Path(base.__file__), Path(__file__).with_name('backend_error_metrics.py'),
                                       *[REFERENCE/n for n in ('d1_methods.py','n2_model.py','prepare.py','n2_config.json')]]})
        for method, a in arrays.items():
            bits = np.unpackbits(np.bitwise_xor(arrays['n3'].view(np.uint8), a.view(np.uint8))).sum()
            entry = dict(logits_vs_n3=numeric(arrays['n3'], a), mismatched_bits=int(bits),
                         logits_physical_vs_n3=numeric(arrays['n3'].astype(np.float64)*self.scale,
                                                      a.astype(np.float64)*self.scale),
                         logit_error_units='logits_vs_n3: signed INT8 code LSB; logits_physical_vs_n3: codes times common frozen head scale',
                         total_logit_bits=int(a.size*8), scopes={})
            for scope, ix in scopes.items():
                entry['scopes'][scope] = dict(
                    integer_vs_n3=paired_labels(predictions['n3'], predictions[method], self.gt, ix, a.shape[1], margins['n3']),
                    integer_vs_qat=paired_labels(self.qat_pred, predictions[method], self.gt, ix, a.shape[1], self.qat_margin),
                    float_interpolation_vs_n3=paired_labels(floats['n3'], floats[method], self.gt, ix, a.shape[1], margins['n3']),
                    integer_vs_float_postprocess_mismatches=int(np.count_nonzero(predictions[method].ravel()[ix]!=floats[method].ravel()[ix])))
            summary['methods'][method] = entry
            np.save(self.out/f'{method}_disagreement_mask_vs_n3.npy', predictions[method] != predictions['n3'])
            t = entry['scopes']['test']['integer_vs_n3']
            print(f'[BACKEND {method}] logit MAE={entry["logits_vs_n3"]["MAE"]:.8g}; '
                  f'test label disagreements={t["mismatches"]}/{t["count"]}; '
                  f'correct->wrong={t["reference_correct_actual_wrong"]}, wrong->correct={t["reference_wrong_actual_correct"]}', flush=True)
        for rows in self.trace.get('isolated', {}).values():
            if any(v['mismatches'] for k, v in rows.items() if '/d_path@' in k):
                raise AssertionError('Same-input isolated replay changed common D')
        for category, methods in self.trace.items():
            dump(self.out/f'{category}_ssm_vs_n3.json', {m:{k:finished(v) for k,v in rows.items()} for m,rows in methods.items()})
        for category, methods in self.float_errors.items():
            dump(self.out/f'{category}_ssm_vs_float.json', {m:{k:finished(v) for k,v in rows.items()} for m,rows in methods.items()})
        dump(self.out/'layer_propagation.json', {m:{k:finished(v) for k,v in rows.items()} for m,rows in self.layers.items()})
        dump(self.out/'layer_fixed_vs_float.json', {m:{k:finished(v) for k,v in rows.items()} for m,rows in self.layer_float.items()})
        dump(self.out/'tile_manifest.json', self.placements)
        # Written last. The runner additionally binds inputs, source hashes and output hashes.
        dump(self.out/'accuracy_propagation_summary.json', summary)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--qat-run-dir', required=True)
    p.add_argument('--fp32-dir', required=True)
    p.add_argument('--dataset', choices=['UP','HanChuan','HongHu','Houston'], required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--data-path', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--trace-tiles', type=int, default=5)
    a = p.parse_args()
    if a.trace_tiles < -1:
        p.error('trace-tiles must be >=0 or -1')
    qat = json.loads((Path(a.qat_run_dir)/'result.json').read_text(encoding='utf-8'))
    from run_fpga_qat_eval1_four_datasets import validate_qat
    validate_qat(qat, a.qat_run_dir)
    config = SSMNumericConfig.from_dict(qat.get('numeric_config'))
    if config.resolved_readout_requantization != 'hardware':
        raise ValueError('Hardware C+D requantization required')
    ex = Experiment(a.trace_tiles)
    ex.out = Path(a.output_dir)/'nonlinear_experiment'
    ex.install()
    try:
        result = base.simulate_mamba_hsi_dense16_both(
            qat_run_dir=a.qat_run_dir, dataset_name=a.dataset, seed=a.seed,
            fp32_dir=a.fp32_dir, data_path=a.data_path, device_name=a.device,
            output_dir=a.output_dir, numeric_config=config, qat_reference_batch_size=1)
        if result['saved_qat_prediction_match'] != 1 or result['qat_batch1_vs_saved_batch_match'] != 1:
            raise ValueError('QAT eval1 reference reproduction failed')
        # Correct obsolete descriptive text only; no arithmetic modifications.
        result['simulation_scope'] = 'D1 eval1 integer baseline; N0-N3 paired backend study; new checkpoints are not claimed RTL-validated'
        dump(Path(a.output_dir)/'fpga_simulation_result.json', result)
        ex.finish(result)
    finally:
        ex.restore()


if __name__ == '__main__':
    main()
