"""CPU-only ordering regression plus optional real CUDA:1 restoration test."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import torch
import analyze_alog_dynamics as analyzer
import train_mambahsi_spatial_split_dense_qat as qat


class RestoreDeviceTests(unittest.TestCase):
    def test_prepare_and_fusion_tensors_moved_before_use(self):
        # Simulate constructor-created CPU LSQ parameters and a replacement
        # buffer from fusion, without needing a physical CUDA device locally.
        class Model:
            def __init__(self): self.tensors={'weight':torch.device('cpu')}
            def to(self, device):
                self.tensors={k:torch.device(device) for k in self.tensors}
                return self
            def eval(self): return self
            def named_parameters(self):
                return [(k,SimpleNamespace(device=v)) for k,v in self.tensors.items() if k!='fold_buffer']
            def named_buffers(self):
                return [(k,SimpleNamespace(device=v)) for k,v in self.tensors.items() if k=='fold_buffer']
            def load_state_dict(self, state, strict):
                self_test.assertTrue(strict)
                self_test.assertTrue(all(v==torch.device('cuda:1') for v in self.tensors.values()))
        self_test=self
        model=Model()
        def prepare(m, **kwargs): m.tensors['lsq_a.s']=torch.device('cpu')
        def fold(m): m.tensors['fold_buffer']=torch.device('cpu')
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp);run=source/'run_seed0';run.mkdir()
            (run/'best_qat_foldaware.pth').touch()
            (source/'spatial_split.json').write_text('{}')
            (run/'result.json').write_text(json.dumps(dict(dataset='UP',seed=0,model_config={})))
            (run/'model_config.json').write_text(json.dumps(dict(in_channels=16,num_classes=9)))
            with patch.object(qat,'build_configured_model',return_value=model), \
                 patch.object(qat,'prepare_qat_model',side_effect=prepare), \
                 patch.object(qat,'freeze_lsq_initialization'), \
                 patch.object(qat,'fuse_qat_model_bns_for_deploy',side_effect=fold), \
                 patch.object(analyzer,'load_state_dict',return_value={}):
                restored,_=analyzer.restore_model(run,torch.device('cuda:1'),artifact_dir=source)
            analyzer.assert_model_device(restored,'cuda:1')
            model.tensors['lsq_a.s']=torch.device('cpu')
            with self.assertRaisesRegex(RuntimeError,'lsq_a.s=cpu'):
                analyzer.assert_model_device(model,'cuda:1')

    def test_registered_buffer_guard(self):
        m=torch.nn.Linear(2,2)
        m.register_buffer('stray',torch.empty(1,device='meta'))
        with self.assertRaisesRegex(RuntimeError,'stray=meta'):
            analyzer.assert_model_device(m,'cpu')


if __name__=='__main__': unittest.main()
