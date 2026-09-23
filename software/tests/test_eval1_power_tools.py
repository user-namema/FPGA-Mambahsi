import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import gpu_power_monitor as power
import run_qat_eval_batch1_four_datasets as qat_run


class PowerTests(unittest.TestCase):
    def test_clipped_linear_power_integral(self):
        samples=[dict(monotonic_s=0,power_w=100),dict(monotonic_s=10,power_w=200)]
        self.assertAlmostEqual(power.integrate_power(samples,2,8),900)

    def test_total_energy_not_idle_subtracted_and_units(self):
        samples=[dict(monotonic_s=0,power_w=100),dict(monotonic_s=10,power_w=100)]
        r=power.energy_summary(samples,0,10,2000,3100,100)
        self.assertEqual(r['energy_method'],'nvml_energy_counter')
        self.assertEqual(r['mean_power_w'],110)
        self.assertEqual(r['joules_per_tile'],11)
        self.assertEqual(r['microjoules_per_tile'],11000000)
        self.assertEqual(r['power_integrated_energy_j'],1000)

    def test_unsupported_or_stuck_energy_falls_back(self):
        samples=[dict(monotonic_s=0,power_w=100),dict(monotonic_s=10,power_w=100)]
        for e0,e1 in [(None,None),(4,4),(4,3)]:
            r=power.energy_summary(samples,0,10,e0,e1,100)
            self.assertEqual(r['energy_method'],'nvml_power_trapezoid')
            self.assertEqual(r['energy_j'],1000)

    def test_empty_or_negative_data_rejected(self):
        with self.assertRaises(ValueError):power.integrate_power([],0,1)
        with self.assertRaises(ValueError):power.integrate_power([dict(monotonic_s=0,power_w=-1)],0,1)

    def test_device_mapping_uses_runtime_pci_not_cuda_index(self):
        class Error(Exception):pass
        fake=SimpleNamespace(NVMLError=Error,nvmlInit=lambda:None,
            nvmlDeviceGetHandleByPciBusId=lambda bus:('pci',bus),
            nvmlDeviceGetUUID=lambda h:'GPU-selected',nvmlDeviceGetName=lambda h:'RTX',
            nvmlSystemGetDriverVersion=lambda:'test',nvmlDeviceGetPowerManagementLimit=lambda h:450000)
        with patch.dict('sys.modules',{'pynvml':fake}),patch.object(power,'cuda_pci_bus_id',return_value='0000:65:00.0') as pci,patch.object(power.NVMLDevice,'read',return_value={}):
            d=power.NVMLDevice(0)
            self.assertEqual(d.handle,('pci',b'0000:65:00.0'))
            pci.assert_called_once_with(0)


class QATTests(unittest.TestCase):
    def test_launcher_accepted_by_actual_training_parser(self):
        import train_mambahsi_spatial_split_dense_qat as q
        a=SimpleNamespace(data_path='./data',seeds='0,6',device='cuda:1',max_epoch=100)
        cmd=qat_run.build_command(Path('train.py'),'UP',Path('/tmp/fp32'),Path('/tmp/result'),a)
        args=q.build_parser().parse_args(cmd[3:]);q.resolve_model_arguments(args);q.validate_args(args)
        self.assertEqual((args.batch_size,args.eval_batch_size),(32,1))
        self.assertEqual((args.freeze_bn_epoch,args.freeze_lsq_epoch),(20,20))
        self.assertEqual(args.d_init_policy,'mean');self.assertEqual(args.weight_init_policy,'patch_max')
        self.assertTrue(args.use_D);self.assertFalse(args.use_z)

    def test_real_loaders_and_evaluation_use_one_tile(self):
        import numpy as np
        import torch
        import train_mambahsi_spatial_split_dense_qat as q
        image=np.ones((16,48,16),dtype=np.float32)
        labels=[np.zeros((16,48),dtype=np.int64) for _ in range(3)]
        labels[2][:,16:32]=-1  # middle tile is outside the synthetic test split
        tiles=[[(0,16,0,16,0),(0,16,16,32,0)],[(0,16,32,48,1)],[(0,16,0,16,2),(0,16,32,48,2)]]
        args=SimpleNamespace(batch_size=32,eval_batch_size=1,num_workers=0,tile_size=16)
        _,loaders=q.make_seed_datasets_loaders(args,image,labels,tiles,0)
        self.assertEqual([l.batch_size for l in loaders],[32,1,1])
        class Model(torch.nn.Module):
            def __init__(self):super().__init__();self.sizes=[]
            def forward(self,x):
                self.sizes.append(x.shape[0]);return torch.ones((x.shape[0],1,4,4))
        model=Model()
        q.evaluate_model(model,loaders[2],(16,48),16,torch.device('cpu'),labels[2],1)
        self.assertEqual(model.sizes,[1,1])
        self.assertFalse(model.training)

    def test_result_audit_rejects_old_batch8(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);d=root/'models/x/run_seed0';d.mkdir(parents=True)
            r=dict(dataset='UP',seed=0,batch_size=32,eval_batch_size=8,
                best_validation_recheck_match=True,bn_fold_test_prediction_match=1.,
                best_epoch=20,fp32_test_OA=.9,qat_test_OA=.91,qat_test_mAcc=.9)
            p=d/'result.json';p.write_text(json.dumps(r))
            with self.assertRaises(RuntimeError):qat_run.validate_results(root,[0])
            r['eval_batch_size']=1;p.write_text(json.dumps(r))
            self.assertEqual(qat_run.validate_results(root,[0])[0]['eval_batch_size'],1)
            with self.assertRaises(RuntimeError):qat_run.validate_results(root,[0,1])

if __name__=='__main__':unittest.main()
