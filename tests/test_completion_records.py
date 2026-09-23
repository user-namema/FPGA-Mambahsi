import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock
ROOT=Path(__file__).resolve().parents[1]
def load(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'tools'/(name+'.py'))
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
class CompletionTests(unittest.TestCase):
    def test_published_records(self):
        result=load('verify_completion_records').verify(ROOT,check_arrays=True)
        self.assertEqual(result['fp32_reports'],28)
        self.assertEqual(result['dt_rows'],18)
    def test_dry_run_maps_device_without_writes_or_execution(self):
        mod=load('run_nonlinear_d1');output=io.StringIO()
        with mock.patch.dict(mod.os.environ,{'CUDA_VISIBLE_DEVICES':'3,5'},clear=True), mock.patch.object(mod.subprocess,'run') as run, mock.patch.object(Path,'mkdir',side_effect=AssertionError('write')),contextlib.redirect_stdout(output):
            mod.main(['--qat-run-dir','/tmp/model','--device','cuda:1','--dry-run'])
        run.assert_not_called()
        self.assertIn('CUDA_VISIBLE_DEVICES=5',output.getvalue())
        self.assertIn('--nonlinear-backend all',output.getvalue())
    def test_wrong_checkpoint_rejected_before_child(self):
        mod=load('run_nonlinear_d1')
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder);(path/'best_qat_foldaware.pth').write_bytes(b'wrong checkpoint')
            with mock.patch.object(mod.subprocess,'run') as run,contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError,'Wrong checkpoint'):
                    mod.main(['--qat-run-dir',str(path),'--device','cpu','--output-dir',str(path/'new')])
                run.assert_not_called()
            self.assertFalse((path/'new').exists())
if __name__=='__main__':unittest.main()
