"""Runner tests use fake metadata; they do NOT claim checkpoint inference."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_backend_error_four_datasets as runner


class RunnerTests(unittest.TestCase):
    def fixture(self, root):
        qat=root/'qat'/'UP'/'run_seed0'; qat.mkdir(parents=True)
        fp=root/'fp'; (fp/'run_seed0').mkdir(parents=True)
        data=root/'data'; data.mkdir()
        (data/'UP.mat').write_bytes(b'synthetic test fixture, not an actual dataset')
        config=dict(dt_input_bits=9,dt_output_bits=8,a_fraction_bits=24,k_bits=19,
                    state_bits=32,state_fraction_bits=24,rounding='single',coefficient_backend='exact',error_source='all')
        meta=dict(dataset='UP',seed=0,batch_size=32,eval_batch_size=1,weight_init_policy='patch_max',
                  d_init_policy='mean',freeze_bn_epoch=20,freeze_lsq_epoch=20,
                  best_validation_recheck_match=True,bn_fold_test_prediction_match=1,
                  model_config=dict(use_D=True),numeric_config=config,source_fp32_dir=str(fp))
        (qat/'result.json').write_text(json.dumps(meta))
        for n in ('best_qat_foldaware.pth','sample_indices.npz','qat_deploy_test_prediction.npy'):
            (qat/n).write_bytes(b'test')
        for n in ('train_only_preprocess.npz','spatial_split_masks.npz','spatial_split.json','run_seed0/sample_indices.npz'):
            (fp/n).write_bytes(b'test')
        return ['--qat-root',str(root/'qat'),'--data-path',str(data),'--datasets','UP','--seeds','0',
                '--device','cpu','--output-dir',str(root/'out')], qat

    def test_preflight_never_launches_or_creates_output(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); args,qat=self.fixture(root)
            with patch.object(runner.subprocess,'run') as launch, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(runner.main(args+['--dry-run']),0)
                launch.assert_not_called()
            self.assertFalse((root/'out').exists())
            (qat/'sample_indices.npz').unlink()
            with self.assertRaises(FileNotFoundError): runner.main(args+['--dry-run'])

    def test_resume_keeps_failed_attempt_and_rejects_changed_input(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); args,qat=self.fixture(root)
            with patch.object(runner.subprocess,'run') as launch, contextlib.redirect_stdout(io.StringIO()):
                launch.return_value.returncode=9
                with self.assertRaises(RuntimeError): runner.main(args)
                with self.assertRaises(RuntimeError): runner.main(args+['--resume'])
                self.assertTrue((root/'out/UP/seed0/attempt001').is_dir())
                self.assertTrue((root/'out/UP/seed0/attempt002').is_dir())
                self.assertEqual(launch.call_count,2)
                (qat/'best_qat_foldaware.pth').write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError,'Resume inputs'): runner.main(args+['--resume'])
                self.assertEqual(launch.call_count,2)

    def test_rejects_d0_and_wrong_numeric(self):
        with tempfile.TemporaryDirectory() as d:
            args,qat=self.fixture(Path(d))
            meta=json.loads((qat/'result.json').read_text())
            meta['model_config']['use_D']=False
            (qat/'result.json').write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError,'D1'): runner.main(args+['--dry-run'])
            meta['model_config']['use_D']=True
            meta['numeric_config']['dt_output_bits']=7
            (qat/'result.json').write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError,'contract'): runner.main(args+['--dry-run'])


if __name__=='__main__': unittest.main()
