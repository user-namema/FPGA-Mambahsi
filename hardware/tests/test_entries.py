"""Host execution checks for the nonlinear and power command entries."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class NonlinearEntryTests(unittest.TestCase):
    def interpreter(self):
        try:
            import tkinter
            return tkinter.Tcl()
        except Exception as exc:
            self.skipTest('Tcl runtime unavailable: ' + str(exc))

    def dispatch(self, *args):
        interp = self.interpreter()
        interp.setvar('argc', len(args))
        interp.setvar('argv', interp.call('list', *args))
        interp.eval('proc source {path} {set ::selected_entry [file tail $path]}')
        interp.eval((ROOT / 'nonlinear/scripts/dispatch.tcl').read_text(encoding='utf8'))
        return interp

    def test_dispatch_requires_output(self):
        for args in [('sim', 'N3', 'blk0_spa', '1'),
                     ('sim', 'N3', 'blk0_spa', '1', '')]:
            with self.subTest(args=args):
                with self.assertRaisesRegex(Exception, 'new_out'):
                    self.dispatch(*args)

    def test_dispatch_forwards_explicit_output(self):
        for action, entry in [('sim', 'run_sim.tcl'), ('d-sim', 'run_d1_sim.tcl'),
                              ('build', 'build_ooc.tcl')]:
            with self.subTest(action=action):
                interp = self.dispatch(action, 'N3', 'blk0_spa', '1', 'D:/fpga/custom_ooc')
                self.assertEqual(interp.getvar('nl_out'), 'D:/fpga/custom_ooc')
                self.assertEqual(interp.getvar('selected_entry'), entry)

    def test_missing_package_error_identifies_restoration_location(self):
        interp = self.interpreter()
        with tempfile.TemporaryDirectory(prefix='nl_entry_') as name:
            script = Path(name) / 'scripts/common.tcl'
            script.parent.mkdir()
            script.write_text((ROOT / 'nonlinear/scripts/common.tcl').read_text(encoding='utf8'),
                              encoding='utf8')
            with self.assertRaisesRegex(Exception, 'hardware/nonlinear') as caught:
                interp.call('source', script.as_posix())
            self.assertNotIn('tools/prepare.py', str(caught.exception))


class PowerTclTests(unittest.TestCase):
    def interpreter(self):
        try:
            import tkinter
            interp = tkinter.Tcl()
            interp.call('encoding', 'system', 'utf-8')
            return interp
        except Exception as exc:
            self.skipTest('Tcl runtime unavailable: ' + str(exc))

    def setup_build(self, build, interp):
        scripts = build / 'package/scripts'
        scripts.mkdir(parents=True)
        (scripts / 'common.tcl').write_text(
            (ROOT / 'scripts/common.tcl').read_text(encoding='utf8'), encoding='utf8')
        config = {
            'nf_out': build.as_posix(),
            'nf_package': (build / 'package').as_posix(),
            'hw_profile': 'N3',
            'nf_method': 'N3',
            'nf_period_ns': '10',
            'nf_source_hash': 'a' * 64,
            'nf_python': 'C:/previous Python/python.exe',
        }
        (build / 'config.tcl').write_text(
            'incr ::config_reads\n' + ''.join(
                'set ' + name + ' {' + value + '}\n' for name, value in config.items()),
            encoding='utf8')
        interp.setvar('config_reads', 0)
        interp.setvar('test_build', build.as_posix())
        interp.setvar('test_hash', config['nf_source_hash'])
        interp.eval('''
            set exec_calls {}
            proc exec {args} {lappend ::exec_calls $args; return $::test_hash}
            proc set_param {args} {}
            proc get_projects {args} {return {}}
            proc open_project {args} {}
            proc current_project {} {return test_project}
            proc get_property {name object} {return $::test_build}
            proc current_run {args} {}
            proc get_runs {name} {return $name}
            proc close_sim {args} {}
            proc create_fileset {args} {error POWER_TEST_AFTER_CONFIG_RELOAD}
        ''')
        return config

    def test_current_python_survives_both_power_config_reads(self):
        interp = self.interpreter()
        with tempfile.TemporaryDirectory(prefix='power_entry_', dir=ROOT.parent) as name:
            build = Path(name)
            self.setup_build(build, interp)
            selected_python = 'C:/selected Python/python.exe'
            args = ('probe', build.as_posix(), (build / 'output').as_posix(),
                    (build / 'output').as_posix(), selected_python)
            interp.setvar('argv', interp.call('list', *args))
            with self.assertRaisesRegex(Exception, 'POWER_TEST_AFTER_CONFIG_RELOAD'):
                interp.call('source', (ROOT / 'power/n3_power.tcl').as_posix())
            self.assertEqual(int(interp.getvar('config_reads')), 2)
            self.assertEqual(interp.getvar('nf_python'), selected_python)
            commands = [interp.splitlist(command)
                        for command in interp.splitlist(interp.getvar('exec_calls'))]
            self.assertGreaterEqual(len(commands), 5)
            self.assertTrue(all(command[0] == selected_python for command in commands), commands)
            verification = [command for command in commands if command[2].endswith('/tools/verify.py')]
            self.assertEqual(len(verification), 2)

    def test_build_loader_uses_config_python_without_override(self):
        interp = self.interpreter()
        with tempfile.TemporaryDirectory(prefix='power_default_', dir=ROOT.parent) as name:
            build = Path(name)
            config = self.setup_build(build, interp)
            interp.setvar('nf_out', build.as_posix())
            interp.call('source', (build / 'package/scripts/common.tcl').as_posix())
            interp.call('nf_load_build')
            self.assertEqual(interp.getvar('nf_python'), config['nf_python'])
            self.assertEqual(int(interp.getvar('config_reads')), 1)
            commands = interp.splitlist(interp.getvar('exec_calls'))
            self.assertEqual(interp.splitlist(commands[0])[0], config['nf_python'])


@unittest.skipUnless(os.name == 'nt', 'Windows batch entries')
class PowerBatchTests(unittest.TestCase):
    def launch(self, entry, *args):
        env = dict(os.environ, NF_PYTHON=sys.executable)
        return subprocess.run([str(ROOT / 'power' / entry), *args], env=env,
                              capture_output=True, text=True, errors='replace', timeout=30)

    def test_entries_forward_help(self):
        for entry in ('run_n3_power.bat', 'run_saif_probe.bat', 'run_capture_after_probe.bat'):
            with self.subTest(entry=entry):
                result = self.launch(entry, '--help')
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn('--build', result.stdout)
                self.assertIn('--vivado', result.stdout)
                self.assertIn('--probe-dir', result.stdout)

    def test_entries_preserve_failure_exit(self):
        for entry in ('run_n3_power.bat', 'run_saif_probe.bat', 'run_capture_after_probe.bat'):
            with self.subTest(entry=entry):
                result = self.launch(entry)
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertIn('--build', result.stderr)


if __name__ == '__main__':
    unittest.main(verbosity=2)
