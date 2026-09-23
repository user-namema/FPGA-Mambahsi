import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT=Path(__file__).resolve().parents[1]/'run_fpga_qat_eval1_four_datasets.py'
spec=importlib.util.spec_from_file_location('runner',SCRIPT)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class RunnerTests(unittest.TestCase):
    def test_device_mapping(self):
        self.assertEqual(m.device_environment('cuda:1',{})[1]['CUDA_VISIBLE_DEVICES'],'1')
        self.assertEqual(m.device_environment('cuda:1',{'CUDA_VISIBLE_DEVICES':'2,5'})[1]['CUDA_VISIBLE_DEVICES'],'5')
        self.assertEqual(m.device_environment('cuda',{'CUDA_VISIBLE_DEVICES':'2,5'})[1]['CUDA_VISIBLE_DEVICES'],'2,5')
        with self.assertRaises(ValueError):m.device_environment('cuda:1',{'CUDA_VISIBLE_DEVICES':'2'})

    def test_full_launch_failure_resume_and_changed_input(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);project=root/'project';project.mkdir();run=root/'qat/UP/models/run_seed0';run.mkdir(parents=True)
            fp=root/'fp';(fp/'run_seed0').mkdir(parents=True);data=root/'data';data.mkdir()
            for name in ['train_only_preprocess.npz','spatial_split_masks.npz','spatial_split.json','run_seed0/sample_indices.npz']:(fp/name).write_bytes(b'fixture')
            for name in ['best_qat_foldaware.pth','sample_indices.npz','qat_deploy_test_prediction.npy']:(run/name).write_bytes(b'fixture')
            numeric=dict(error_source='all',readout_requantization='auto')
            result=dict(dataset='UP',seed=0,batch_size=32,eval_batch_size=1,weight_init_policy='patch_max',d_init_policy='mean',freeze_bn_epoch=20,freeze_lsq_epoch=20,best_validation_recheck_match=True,bn_fold_test_prediction_match=1,model_config={'use_D':True},numeric_config=numeric,source_fp32_dir=str(fp))
            (run/'result.json').write_text(json.dumps(result))
            stub='''import sys,json,hashlib
from pathlib import Path
args=dict(zip(sys.argv[1::2],sys.argv[2::2])) if False else None
def val(k):return sys.argv[sys.argv.index(k)+1]
p=Path(val('--output-dir'));run=Path(val('--qat-run-dir'))
if (Path.cwd()/'fail_once').exists():
 (Path.cwd()/'fail_once').unlink();raise SystemExit(7)
metric={'OA':.9,'mAcc':.8,'Kappa':.85}
r=dict(dataset='UP',seed=0,qat_reference_batch_size=1,saved_qat_prediction_match=1,qat_batch1_vs_saved_batch_match=1,qat_best_epoch=5,qat_checkpoint_sha256=hashlib.sha256((run/'best_qat_foldaware.pth').read_bytes()).hexdigest(),numeric_config=json.loads((run/'result.json').read_text())['numeric_config'],ssm_execution_mode='exact-integer',metrics={k:metric for k in ['qat_saved_batch','qat_direct','staged_fp','int8_hw']},test_agreements={'qat_direct_vs_int8':{'rate':.98}})
(p/'fpga_simulation_result.json').write_text(json.dumps(r))
'''
            for name in m.SOURCES:(project/name).write_text(stub if name==m.SOURCES[0] else '# fixture')
            cmd=[sys.executable,str(SCRIPT),'--project-root',str(project),'--qat-root',str(root/'qat'),'--data-path',str(data),'--output-dir',str(root/'out'),'--datasets','UP','--seeds','0','--device','cpu']
            def call(extra=()):return subprocess.run(cmd+list(extra),capture_output=True,text=True)
            dry=call(['--dry-run']);self.assertEqual(dry.returncode,0,dry.stderr);self.assertFalse((root/'out').exists())
            (project/'fail_once').touch();self.assertNotEqual(call().returncode,0)
            retry=call(['--resume']);self.assertEqual(retry.returncode,0,retry.stderr)
            self.assertTrue((root/'out/UP/seed0/attempt001').exists());self.assertTrue((root/'out/UP/seed0/attempt002/fpga_simulation_result.json').exists())
            self.assertTrue((root/'out/per_seed_summary.csv').is_file())
            skipped=call(['--resume']);self.assertEqual(skipped.returncode,0,skipped.stderr);self.assertIn('Skip complete',skipped.stdout)
            (run/'best_qat_foldaware.pth').write_bytes(b'changed');self.assertNotEqual(call(['--resume']).returncode,0)

if __name__=='__main__':unittest.main()
