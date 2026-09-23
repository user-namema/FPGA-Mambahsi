"""Full-network N0/N1/N2/N3 comparison on the validated D1 numerical graph.

Run from this directory, retaining the supplied QAT/model/helper modules.
Only Abar/K are replaced. Native D1 scan_codes performs the recurrence and
C+D addition, and the native simulator performs the single output requant.
"""
import argparse
import contextlib
import io
import inspect
import sys
from pathlib import Path
import numpy as np
import torch
import both_FPGA_single_qat_source as base
from ssm_error_ablation import SSMNumericConfig
from d1_methods import CHECKPOINT, INPUT_SHA256, LABEL_SHA256, CORES, FIT, constants, variants, export_tables, digest, dump, stats, merge, finished
from d1_methods import check_coefficient_files


class Recorder:
    def __init__(self, experiment, method, category, reference, downstream=None):
        self.experiment, self.method, self.category = experiment, method, category
        self.reference, self.downstream = reference, downstream

    def add(self, core, time, actual, reference, saturation=0):
        if self.downstream is not None:
            self.downstream.add(core, time, actual, reference, saturation)
        ex = self.experiment
        key = (core, int(time))
        # Values here are physical values from the UNMODIFIED native D1 recorder.
        # State is Q24 / 2^24; C/D/total also include the frozen s_C.
        if self.method == 'n3':
            self.reference[key] = actual.detach().cpu().clone()
        else:
            row = ex.trace.setdefault(self.category, {}).setdefault(self.method, {}).setdefault(f'{core}@{time}', {})
            merge(row, stats(self.reference[key], actual))
            row['saturations'] = row.get('saturations', 0) + int(saturation)
        local = ex.float_errors.setdefault(self.category, {}).setdefault(self.method, {}).setdefault(f'{core}@{time}', {})
        merge(local, stats(reference, actual))


class Experiment:
    def __init__(self, args):
        self.args = args
        self.methods = ['n0', 'n1', 'n2'] if args.nonlinear_backend == 'all' else (
            [] if args.nonlinear_backend == 'n3' else [args.nonlinear_backend])
        self.method, self.tile = 'n3', -1
        self.tables, self.parameters, self.table_names = {}, {}, {}
        self.logits = {m: [] for m in ['n3', *self.methods]}
        self.layers, self.trace, self.float_errors = {}, {}, {}
        self.layer_float = {}
        self.boundaries, self.states = {}, {}
        self.placements = None
        self.gt = self.indices = None
        self.original_scan = base.scan_codes
        self.original_tile = base.simulate_tile_dual_int8
        self.original_luts = base.get_or_build_ssm_luts

    def traced(self):
        return self.args.trace_tiles < 0 or self.tile < self.args.trace_tiles

    def coefficients(self, m, sdt, sb, su, name, output_dir, device):
        result = self.original_luts(m, sdt, sb, su, name,
                                    output_dir if self.method == 'n3' else None, device)
        if not getattr(m, 'use_D', False):
            raise ValueError(f'{name}: D is disabled; refusing a mixed D0/D1 experiment')
        cache = m.__dict__['_fpga_ssm_lut_cache']
        signature = cache['content_signature_sha256']
        if name in self.parameters and self.parameters[name]['signature'] != signature:
            raise RuntimeError(f'{name}: frozen coefficient contract changed')
        if name not in self.tables:
            p = constants(m.A_log_shared, cache['s_dt'], cache['s_b'], cache['s_u'], result[3])
            p['signature'] = signature
            self.parameters[name] = p
            self.tables[name] = variants(cache['tables'], p, device)
        self.table_names[id(cache['tables'])] = name
        # scan_codes below selects the table. Returning a different tuple alone
        # would NOT change D1, whose native simulator reads cache['tables'].
        return result

    def scan(self, u, dt, b, readout, tables, config=None, scales=(1.,1.),
             recorder=None, core='', d_tables=None, components=None):
        if d_tables is None:
            raise ValueError(f'{core}: missing D table')
        if self.table_names.get(id(tables)) != core:
            raise ValueError(f'{core}: coefficient cache identity mismatch')
        if 'd_coefficient' not in self.parameters[core]:
            self.parameters[core] = export_tables(self.out/'coefficients', core,
                self.tables[core], self.parameters[core], d_tables)
            anchor = Path(__file__).parent/'hardware_reference'
            if not anchor.is_dir():
                raise FileNotFoundError('Copy the whole server directory, including hardware_reference')
            import json
            expected = json.loads((anchor/f'{core}_parameters.json').read_text(encoding='utf-8'))
            for field in ('SDT_Q30','SBSU_Q40','DECAY_Q24','K_fraction_bits','d_coefficient','d_left_shift'):
                if self.parameters[core][field] != expected[field]:
                    raise AssertionError(f'{core}/{field}: software differs from packaged D1 hardware')
            for method in ('n0','n1','n2','n3'):
                check_coefficient_files(self.out/f'coefficients/{core}_{method}.mem',
                                        anchor/f'{core}_{method}.mem',f'{core}/{method}')
            print(f'[D1 COEFFICIENTS OK] {core}: N0/N1/N2/N3 x 256 addresses x 419 bits; D constants exact')
        p = self.parameters[core]
        if (p['d_coefficient'] != d_tables['coefficient'].cpu().tolist()
                or p['d_left_shift'] != d_tables['left_shift']):
            raise ValueError(f'{core}: D changed between methods/tiles')
        rec = Recorder(self, self.method, 'end_to_end', self.states,
                       recorder if self.method == 'n3' else None) if self.traced() else (
                       recorder if self.method == 'n3' else None)
        result = self.original_scan(u, dt, b, readout, self.tables[core][self.method], config,
                    scales=scales, recorder=rec, core=core, d_tables=d_tables,
                    components=components if self.method == 'n3' else None)
        if self.method == 'n3' and self.traced():
            # Replay on identical N3 U/dt/B/C; the SAME compiled D is used once.
            # Thus isolated D error vs N3 must be zero, but end-to-end D may
            # differ because upstream nonlinear substitutions can change U.
            for method in self.methods:
                self.original_scan(u, dt, b, readout, self.tables[core][method], config,
                    scales=scales, recorder=Recorder(self, method, 'isolated', self.states),
                    core=core, d_tables=d_tables)
        return result

    def tile_run(self, net, tile, scale, output_dir, device):
        self.tile += 1
        self.states.clear()
        self.boundaries.clear()
        self.method = 'n3'
        result = self.original_tile(net, tile, scale, output_dir, device)
        self.logits['n3'].append(result[2][0].cpu().numpy().astype(np.int8))
        capture_flags=[(m,m._capture_ssm_inputs) for m in net.modules() if hasattr(m,'_capture_ssm_inputs')]
        try:
            # The native optional capture uses a module-local output path even
            # when output_dir=None. Never overwrite an N3 capture with a variant.
            for module,_ in capture_flags:
                module._capture_ssm_inputs=False
            for method in self.methods:
                self.method = method
                with contextlib.redirect_stdout(io.StringIO()):
                    other = self.original_tile(net, tile, scale, None, device)
                if not torch.equal(result[0], other[0]) or not torch.equal(result[1], other[1]):
                    raise AssertionError('QAT/staged FP reference changed between backends')
                if float(result[3]) != float(other[3]):
                    raise AssertionError('Frozen head scale changed')
                self.logits[method].append(other[2][0].cpu().numpy().astype(np.int8))
        finally:
            self.method = 'n3'
            for module,flag in capture_flags:
                module._capture_ssm_inputs=flag
        return result

    def install(self):
        base.get_or_build_ssm_luts = self.coefficients
        base.scan_codes = self.scan
        base.simulate_tile_dual_int8 = self.tile_run
        original_report = base.report_fixed_float
        signature = inspect.signature(original_report)
        def report(*args, **kwargs):
            result = original_report(*args, **kwargs)
            if self.tile < 0:
                return result
            values = signature.bind(*args, **kwargs).arguments
            name = values['name']
            # Names differ between simulator revisions; use the actual contract.
            value = values.get('codes')
            if value is None:
                value = values.get('fixed')
            if value is not None:
                if self.method == 'n3':
                    self.boundaries[name] = torch.as_tensor(value).detach().cpu().clone()
                else:
                    row = self.layers.setdefault(self.method, {}).setdefault(name, {})
                    merge(row, stats(self.boundaries[name], value))
                physical = values.get('fixed')
                if physical is None:
                    physical = torch.as_tensor(value).double()*torch.as_tensor(values['scale']).double()
                row = self.layer_float.setdefault(self.method, {}).setdefault(name, {})
                merge(row, stats(values['reference'], physical))
            return result
        base.report_fixed_float = report
        original_export = base.export_rtl_integer_reference
        def export(*args, **kwargs):
            bound = inspect.signature(original_export).bind(*args, **kwargs).arguments
            self.placements = bound['placements']
            self.n3_labels = bound['labels'].copy()
            return original_export(*args, **kwargs)
        base.export_rtl_integer_reference = export
        original_eval = base.evaluate_predictions
        def evaluate(*args, **kwargs):
            self.gt = np.asarray(args[0])
            self.indices = args[9]
            return original_eval(*args, **kwargs)
        base.evaluate_predictions = evaluate
        original_sim = base.simulate_mamba_hsi_dense16_both
        def simulate(*args, **kwargs):
            params = inspect.signature(original_sim).bind(*args, **kwargs).arguments
            self.out = Path(params['output_dir'])/'nonlinear_experiment'
            # Base creates/checks the output directory, so do not create it yet.
            checkpoint = Path(params.get('model_path') or (Path(params['qat_run_dir'])/'best_qat_foldaware.pth'))
            self.checkpoint_sha = digest(checkpoint)
            if self.checkpoint_sha != CHECKPOINT:
                raise ValueError('Not the validated D1 checkpoint. This package intentionally rejects D0/other weights.')
            if params.get('dataset_name', 'UP') != 'UP':
                raise ValueError('This frozen D1 hardware package is for UP; regenerate a new package for other datasets')
            config = SSMNumericConfig.from_dict(params.get('numeric_config'))
            if config.error_source != 'all':
                raise ValueError('Use --ssm-error-source all; no floating counterfactual in N0-N3 comparison')
            result = original_sim(*args, **kwargs)
            # Native snapshot predates the D1 board signoff; correct only its
            # obsolete descriptive text, not any arithmetic/reference tensors.
            result['simulation_scope'] = 'D1 integer network, C+D before one output requantization; N0-N3 comparison uses the frozen D1 graph'
            result['d1_reference_status'] = 'Existing D1 board anchor checked by nonlinear_experiment; new N0-N2 RTL results pending user runs'
            dump(Path(params['output_dir'])/'fpga_simulation_result.json',result)
            self.finish(result)
            return result
        base.simulate_mamba_hsi_dense16_both = simulate

    def finish(self, result):
        if set(self.tables) != set(CORES) or self.placements is None:
            raise AssertionError('Incomplete six-core/full-scene experiment')
        arrays = {m: np.stack(v) for m,v in self.logits.items()}
        count = len(self.placements)
        if any(len(v) != count for v in arrays.values()):
            raise AssertionError('Incomplete tile sequence')
        summary = dict(status='COMPLETE', schema='D1_N0_N3_v1', checkpoint_sha256=self.checkpoint_sha,
            input_sha256=result['rtl_integer_postprocess']['input_tiles_sha256'],
            backends=list(arrays), tiles=count, coefficients=self.parameters, n2_fit=FIT,
            d_rule='native integer C*state + (M_D*U << shift), then one output requantization',
            postprocess='exact integer bilinear score numerators, strict argmax, crop after 16x16 interpolation',
            hardware_simulation_run=False, methods={},
            trace_tiles=min(count,self.args.trace_tiles) if self.args.trace_tiles>=0 else count,
            state_trace_units='physical state; C/D/total in physical readout units (includes s_C)',
            source_sha256={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')},
            frozen_d_equal_across_methods=True)
        truth = self.gt.reshape(-1)
        scopes = dict(all=np.arange(truth.size), labeled=np.flatnonzero(truth>0),
            train=self.indices['train_indices'], validation=self.indices['val_indices'], test=self.indices['test_indices'])
        for method, values in arrays.items():
            pred = np.full(self.gt.shape, -1, dtype=np.int16)
            for tile, place in zip(values, self.placements):
                label, _, _ = base.rtl_integer_bilinear_argmax(tile)
                y,x,h,w = [int(place[k]) for k in ('top','left','valid_height','valid_width')]
                pred[y:y+h,x:x+w] = label[:h,:w]
            if np.any(pred<0):
                raise AssertionError('Uncovered pixels')
            if method == 'n3' and not np.array_equal(pred, self.n3_labels):
                raise AssertionError('N3 wrapper differs from unmodified native postprocess')
            np.save(self.out/f'{method}_tile_logits_int8.npy', values)
            np.save(self.out/f'{method}_prediction_rtl_integer.npy', pred.astype(np.uint8))
            flat = pred.reshape(-1)
            xor=np.bitwise_xor(arrays['n3'].view(np.uint8),values.view(np.uint8))
            bits=sum(bin(i).count('1')*int(n) for i,n in enumerate(np.bincount(xor.reshape(-1),minlength=256)))
            entry = dict(logits_vs_n3=finished(stats(arrays['n3'], values)), agreement={}, metrics={},
                         logit_mismatched_bits=bits,total_logit_bits=int(values.size*8))
            for scope, index in scopes.items():
                ix = np.asarray(index,dtype=np.int64)
                entry['agreement'][scope] = dict(count=len(ix), mismatches=int(np.count_nonzero(flat[ix]!=self.n3_labels.reshape(-1)[ix])))
                if scope != 'all' and len(ix):
                    entry['metrics'][scope] = base.detailed_classification_metrics(truth[ix]-1,flat[ix],values.shape[1])
            summary['methods'][method] = entry
        for method,entry in summary['methods'].items():
            if 'test' in entry['metrics']:
                entry['test_OA_drop_percentage_points']=100*(summary['methods']['n3']['metrics']['test']['OA']-entry['metrics']['test']['OA'])
                print(f"[D1 {method.upper()}] test OA={entry['metrics']['test']['OA']:.8f}; "
                      f"vs N3={entry['test_OA_drop_percentage_points']:.6f} pp")
        for rows in self.trace.get('isolated',{}).values():
            if any(v['mismatches'] for k,v in rows.items() if '/d_path@' in k):
                raise AssertionError('Same-input isolated replay changed the common D path')
        for category, methods in self.trace.items():
            dump(self.out/f'{category}_ssm_vs_n3.json', {m:{k:finished(v) for k,v in rows.items()} for m,rows in methods.items()})
        for category, methods in self.float_errors.items():
            dump(self.out/f'{category}_ssm_vs_float.json', {m:{k:finished(v) for k,v in rows.items()} for m,rows in methods.items()})
        dump(self.out/'layer_propagation.json', {m:{k:finished(v) for k,v in rows.items()} for m,rows in self.layers.items()})
        dump(self.out/'layer_fixed_vs_float.json', {m:{k:finished(v) for k,v in rows.items()} for m,rows in self.layer_float.items()})
        dump(self.out/'tile_manifest.json',self.placements)
        import hashlib
        label_hash=hashlib.sha256(np.ascontiguousarray(self.n3_labels,dtype=np.uint8).tobytes()).hexdigest()
        anchor=dict(input_expected=INPUT_SHA256,input_actual=summary['input_sha256'],
                    labels_expected=LABEL_SHA256,labels_actual=label_hash)
        dump(self.out/'D1_board_anchor_check.json',anchor)
        if summary['input_sha256'] != INPUT_SHA256 or label_hash != LABEL_SHA256:
            raise AssertionError('N3 differs from the validated D1 UP scene input/label hashes; inspect D1_board_anchor_check.json')
        summary['D1_board_input_and_labels_exact']=True
        # Completion marker is written LAST, never on a partial experiment.
        dump(self.out/'accuracy_propagation_summary.json',summary)
        print(f'[D1 N0-N3 COMPLETE] {self.out}')


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--nonlinear-backend', choices=['all','n0','n1','n2','n3'], default='all')
    parser.add_argument('--trace-tiles', type=int, default=5)
    args, rest = parser.parse_known_args()
    if '--help' in rest or '-h' in rest:
        print('Additional D1 options: --nonlinear-backend {all,n0,n1,n2,n3} (default all); '
              '--trace-tiles N (default 5; -1 traces every tile, 0 disables state trace).')
    sys.argv = [sys.argv[0], *rest]
    Experiment(args).install()
    print('[D1 nonlinear v1.1 mem-word-check] Native D1 graph; N0/N1/N2 coefficient replacements; N3 integer baseline.')
    base.main()

if __name__ == '__main__':
    main()
