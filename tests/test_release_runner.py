"""CPU-only release-entry tests: protocol, device mapping and dry-run safety."""
import ast
import contextlib
import csv
import importlib.util
import io
import os
from pathlib import Path
import shlex
import tempfile
import unittest
from unittest import mock


REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('release_runner', REPO/'experiments/run.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def value(command, option):
    return command[command.index(option)+1]


class ReleaseRunnerTests(unittest.TestCase):
    def plan(self, stage, *args, environment=None, root=None):
        stream = io.StringIO()
        with mock.patch.dict(os.environ, environment or {}, clear=True), \
                mock.patch.object(runner, 'ROOT', root or REPO), \
                mock.patch.object(runner.subprocess, 'run') as run, \
                mock.patch.object(Path, 'mkdir', side_effect=AssertionError('dry run wrote a directory')), \
                contextlib.redirect_stdout(stream):
            runner.main([stage, '--dry-run'] + list(args))
            run.assert_not_called()
        jobs = []
        for line in stream.getvalue().splitlines():
            label, command = line.split(': ', 1)
            tokens = shlex.split(command)
            env = {}
            while tokens and '=' in tokens[0] and not tokens[0].startswith('/'):
                key, val = tokens.pop(0).split('=', 1)
                env[key] = val
            jobs.append((label, tokens, env))
        return jobs

    def test_all_nineteen_dry_runs_are_side_effect_free_and_target_valid_clis(self):
        extra = {
            '17_replay_jump': ['--event-dir', './captured/event0'],
            '18_local_ssm': ['--inputs-glob', './captured/*/inputs.npz'],
            '19_batch_diagnosis': ['--prediction-dir', './saved_simulation'],
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for stage in runner.STAGES:
                with self.subTest(stage=stage):
                    jobs = self.plan(stage, *extra.get(stage, []), root=root)
                    self.assertTrue(jobs)
                    for _, command, _ in jobs:
                        if command[0] == 'bash':
                            self.assertTrue(Path(command[1]).is_file())
                            continue
                        script = Path(command[2])
                        self.assertTrue(script.is_file())
                        # Validate emitted flags against the frozen child CLI without
                        # importing Torch/scipy or touching data/checkpoints.
                        options, required = set(), set()
                        for node in ast.walk(ast.parse(script.read_text())):
                            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                                continue
                            if node.func.attr != 'add_argument':
                                continue
                            names = [arg.value for arg in node.args if isinstance(arg, ast.Constant)
                                     and isinstance(arg.value, str) and arg.value.startswith('--')]
                            options.update(names)
                            if any(k.arg == 'required' and isinstance(k.value, ast.Constant)
                                   and k.value.value is True for k in node.keywords):
                                required.add(names[0])
                        emitted = {x for x in command[3:] if x.startswith('--')}
                        self.assertFalse(emitted-options, (stage, emitted-options))
                        self.assertFalse(required-emitted, (stage, required-emitted))
            self.assertEqual(list(root.iterdir()), [])

    def test_paper_architecture_set_has_nineteen_cases_per_dataset(self):
        jobs = self.plan('02_architecture')
        self.assertEqual(len(jobs), 4*19)
        cases = {Path(cmd[1]).stem for _, cmd, _ in jobs}
        self.assertNotIn('04_branch_spe_only', cases)
        self.assertNotIn('03_branch_spa_only', cases)
        self.assertNotIn('11_A_per_channel', cases)
        self.assertIn('21_restore_z_D', cases)

    def test_stability_defaults_cover_six_completed_conditions(self):
        command = self.plan('05_stability')[0][1]
        variants = command[command.index('--variants')+1:command.index('--output-dir')]
        self.assertEqual(len(variants), 6)
        self.assertNotIn('bn1_low_lr_cosine', variants)
        self.assertIn('fixed_low_weight_lr', variants)
        self.assertIn('fixed_freeze_bn_affine', variants)

    def test_eval1_is_one_launcher_call_and_retains_training_batch32(self):
        jobs = self.plan('13_qat_eval1')
        self.assertEqual(len(jobs), 1)
        _, command, env = jobs[0]
        self.assertEqual(command[command.index('--datasets')+1:command.index('--seeds')], runner.DATASETS)
        for ds in runner.DATASETS:
            self.assertIn('FP32_'+ds.upper(), env)
        # Inspect the actual frozen launcher's command expansion, not a duplicate.
        child_spec = importlib.util.spec_from_file_location('eval1_launcher', Path(command[2]))
        child = importlib.util.module_from_spec(child_spec)
        child_spec.loader.exec_module(child)
        from types import SimpleNamespace
        expanded = child.build_command('qat.py', 'UP', '/fp32', Path('/out'),
            SimpleNamespace(data_path='/data', seeds='0', device='cuda:1', max_epoch=100))
        self.assertEqual(value(expanded, '--batch_size'), '32')
        self.assertEqual(value(expanded, '--eval_batch_size'), '1')
        self.assertEqual(value(expanded, '--freeze_bn_epoch'), '20')
        self.assertIn('--include_calibrated_baseline', expanded)
        historical = self.plan('08_qat_eval8', '--datasets', 'UP', '--seeds', '0')[0][1]
        self.assertEqual(value(historical, '--batch_size'), '32')
        self.assertEqual(value(historical, '--eval_batch_size'), '8')

    def test_default_eval1_simulations_consume_stage13_output(self):
        qat = self.plan('13_qat_eval1')[0][1]
        sim = self.plan('14_fpga_eval1')[0][1]
        probe = self.plan('15_dt_inputs')[0][1]
        self.assertEqual(value(qat, '--output-dir'), value(sim, '--qat-root'))
        self.assertEqual(value(qat, '--output-dir'), value(probe, '--qat-root'))
        self.assertIn('--report-dt-inputs', probe)
        self.assertNotEqual(value(sim, '--output-dir'), value(probe, '--output-dir'))

    def test_cuda_index_maps_only_for_legacy_simulator(self):
        visible = {'CUDA_VISIBLE_DEVICES': 'GPU-first,GPU-second'}
        sim = self.plan('09_fpga_eval8', '--datasets', 'UP', '--seeds', '0',
                        '--device', 'cuda:1', environment=visible)[0]
        self.assertEqual(value(sim[1], '--device'), 'cuda')
        self.assertEqual(sim[2]['CUDA_VISIBLE_DEVICES'], 'GPU-second')
        qat = self.plan('13_qat_eval1', '--device', 'cuda:1', environment=visible)[0]
        self.assertEqual(value(qat[1], '--device'), 'cuda:1')
        self.assertEqual(qat[2]['CUDA_VISIBLE_DEVICES'], 'GPU-first,GPU-second')

    def test_paths_are_resolved_in_callers_working_directory(self):
        original = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp:
            try:
                os.chdir(tmp)
                command = self.plan('14_fpga_eval1', '--qat-root', 'qat', '--fp32-root', 'fp32',
                                    '--data-root', 'data', '--output-root', 'out')[0][1]
                for option, name in [('--qat-root', 'qat'), ('--fp32-root', 'fp32'),
                                     ('--data-path', 'data'), ('--output-dir', 'out')]:
                    self.assertEqual(value(command, option), str((Path(tmp)/name).resolve()))
                replay = self.plan('17_replay_jump', '--event-dir', 'events/one')[0][1]
                self.assertEqual(value(replay, '--event-dir'), str((Path(tmp)/'events/one').resolve()))
                local = self.plan('18_local_ssm', '--inputs-glob', 'captures/*/inputs.npz')[0][1]
                self.assertEqual(value(local, '--inputs-glob'), str((Path(tmp)/'captures/*/inputs.npz').resolve()))
            finally:
                os.chdir(original)

    def test_resume_is_forwarded_or_rejected_without_execution(self):
        for stage in runner.STAGES:
            with self.subTest(stage=stage):
                if stage in runner.RESUMABLE:
                    for _, command, _ in self.plan(stage, '--resume'):
                        self.assertIn('--resume', command)
                else:
                    with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                        self.plan(stage, '--resume')
                    self.assertEqual(raised.exception.code, 2)

    def test_fixed_summary_merges_two_datasets_without_duplicate_headers_or_rows(self):
        fields = ['dataset', 'seed', 'batch_size', 'model_kind', 'scope', 'clock', 'median_ms_per_scene']
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            expected = []
            for ds in ['UP', 'HongHu']:
                path = out/ds/'seed0'/'gpu_batch_summary.csv'
                path.parent.mkdir(parents=True)
                rows = [dict(zip(fields, [ds, '0', '1', 'fixed', scope, clock, '12.5']))
                        for scope in ['model_only', 'full_gpu_pipeline']
                        for clock in ['wall_clock', 'cuda_events']]
                expected.extend(rows)
                with path.open('w', newline='', encoding='utf-8-sig') as stream:
                    writer = csv.DictWriter(stream, fieldnames=fields)
                    writer.writeheader(); writer.writerows(rows)
            target = runner.merge_fixed_summaries(out, ['UP', 'HongHu'], [0])
            with target.open(newline='') as stream:
                self.assertEqual(list(csv.DictReader(stream)), expected)
            self.assertEqual(target.read_text().count(','.join(fields)), 1)
            # Reaggregation replaces the same file; it never appends old records.
            runner.merge_fixed_summaries(out, ['UP', 'HongHu'], [0])
            with target.open(newline='') as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 8)
            original = target.read_bytes()
            source = out/'UP'/'seed0'/'gpu_batch_summary.csv'
            with source.open('a', newline='') as stream:
                csv.DictWriter(stream, fieldnames=fields).writerow(expected[0])
            with self.assertRaisesRegex(ValueError, 'Duplicate GPU summary'):
                runner.merge_fixed_summaries(out, ['UP', 'HongHu'], [0])
            self.assertEqual(target.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
