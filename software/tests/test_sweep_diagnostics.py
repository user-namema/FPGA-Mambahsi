"""Regressions for dataset-directory arguments and hidden subprocess failures."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import run_fpga_error_sweep as sweep
import train_mambahsi_spatial_split_dense_qat as qat


class DatasetPathTests(unittest.TestCase):
    def test_root_and_dataset_directory_resolve_same_pair(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); dataset=root/'UP'; dataset.mkdir()
            (dataset/'PaviaU.mat').touch(); (dataset/'PaviaU_gt.mat').touch()
            raw=np.zeros((3,3,16),dtype=np.float32)
            labels=np.arange(1,10).reshape(3,3)
            def read(path):
                return {'paviaU_gt':labels} if path.name.endswith('_gt.mat') else {'paviaU':raw}
            with patch.object(qat,'_read_mat_arrays',side_effect=read) as reader:
                for base in (root,dataset):
                    image,gt,_=qat.load_dataset('UP',base)
                    np.testing.assert_array_equal(image,raw)
                    np.testing.assert_array_equal(gt,labels)
                self.assertEqual([call.args[0].parent for call in reader.call_args_list], [dataset]*4)

    def test_missing_pair_lists_both_locations(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileNotFoundError) as error:
                qat.load_dataset('UP',folder)
            self.assertIn(str(Path(folder)/'UP/PaviaU.mat'),str(error.exception))
            self.assertIn(str(Path(folder)/'PaviaU.mat'),str(error.exception))


class SweepFailureTests(unittest.TestCase):
    def test_child_traceback_is_shown_and_failure_recorded(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); run=root/'run';run.mkdir()
            (run/'result.json').write_text(json.dumps(dict(dataset='UP',seed=0)))
            out=root/'out'
            def fail(command,stdout,**kwargs):
                stdout.write('FileNotFoundError: missing PaviaU.mat\n')
                return subprocess.CompletedProcess(command,1)
            error=io.StringIO()
            with patch.object(sweep.subprocess,'run',side_effect=fail) as launch:
                with contextlib.redirect_stderr(error),contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaises(SystemExit) as exit_error:
                        sweep.main(['--qat-run-dir',str(run),'--data-path',folder,'--output-dir',str(out)])
                self.assertEqual(launch.call_count,1)
            self.assertEqual(exit_error.exception.code,1)
            self.assertIn('FileNotFoundError: missing PaviaU.mat',error.getvalue())
            jobs=json.loads((out/'sweep_manifest.json').read_text())['jobs']
            self.assertEqual(jobs[0]['status'],'failed')
            self.assertFalse(jobs[0]['completed'])
            self.assertEqual(jobs[0]['returncode'],1)
            self.assertTrue(Path(jobs[0]['log_file']).is_file())


if __name__=='__main__': unittest.main()
