import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import run_current_a_gpu_experiments as runner

class CurrentExperiments(unittest.TestCase):
    def test_dry_run_matrix(self):
        for task,expected in [('train-a',8),('analyze-a',16),('gpu',24)]:
            r=subprocess.run([sys.executable,'run_current_a_gpu_experiments.py','--task',task,'--output-dir','/tmp/not-written','--seeds','0,6','--batch-sizes','1,8,32','--dry-run'],check=True,capture_output=True,text=True)
            self.assertEqual(len(r.stdout.strip().splitlines()),expected)
            self.assertNotIn('21patch',r.stdout)
    def test_paired_statistics_and_protocol_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for mode,values in [('shared',[.9,.8]),('per_channel',[.91,.79])]:
                folder=root/'UP'/mode;folder.mkdir(parents=True)
                (folder/'pair_protocol.json').write_text(json.dumps(dict(source_hashes={'data':'same'},max_epoch=400)))
                for seed,value in enumerate(values):
                    run=folder/('run_seed%d'%seed);run.mkdir()
                    (run/'result.json').write_text(json.dumps(dict(dataset='UP',seed=seed,model_config=dict(A_mode=mode,use_D=True,use_z=False),test_OA=value,test_mAcc=value,test_Kappa=value)))
            result=runner.summarize_pairs(root,['UP'],[0,1])[0]
            self.assertAlmostEqual(result['paired_OA_cost_shared_pp_mean'],0)
            self.assertAlmostEqual(result['shared_OA_sample_std'],7.0710678118654755)
            p=root/'UP/per_channel/pair_protocol.json'
            p.write_text(json.dumps(dict(source_hashes={'data':'different'},max_epoch=400)))
            with self.assertRaises(ValueError):runner.summarize_pairs(root,['UP'],[0,1])

if __name__=='__main__':unittest.main()
