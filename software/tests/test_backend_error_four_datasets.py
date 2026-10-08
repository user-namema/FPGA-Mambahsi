import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
import torch.nn.functional as F
import backend_error_four_datasets as worker
from backend_error_metrics import integer_scores, paired_labels, scene_from_tiles
import train_mambahsi_spatial_split_dense_qat as qat
from ssm_error_ablation import SSMNumericConfig, compile_coefficients
from run_backend_error_four_datasets import verify_output


class MetricTests(unittest.TestCase):
    def test_integer_interpolation_and_ties(self):
        rng = np.random.default_rng(91)
        for classes in (9, 16, 22):
            q = rng.integers(-128, 128, size=(classes,4,4),dtype=np.int16)
            actual = integer_scores(q)
            expected = F.interpolate(torch.tensor(q[None],dtype=torch.float64),size=(16,16),mode='bilinear',align_corners=True)[0].numpy()*25
            np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-10)
        self.assertTrue((integer_scores(np.zeros((9,4,4),dtype=np.int8)).argmax(0)==0).all())

    def test_crop_and_coverage(self):
        a = np.zeros((2,3,4,4),dtype=np.int8)
        a[0,1]=3; a[1,2]=4
        places=[dict(top=0,left=0,valid_height=3,valid_width=16),dict(top=0,left=16,valid_height=3,valid_width=1)]
        pred,margin=scene_from_tiles(a,places,(3,17),.2)
        self.assertTrue((pred[:,:16]==1).all())
        self.assertTrue((pred[:,16]==2).all())
        np.testing.assert_allclose(margin[:,16],.8)
        with self.assertRaises(ValueError): scene_from_tiles(a[:1],places[:1],(3,17),.2)
        with self.assertRaises(ValueError): scene_from_tiles(a,[places[0],places[0]],(3,17),.2)

    def test_flips_exclude_unlabeled(self):
        r=np.array([0,0,1,0,1]); a=np.array([1,1,0,1,0]); gt=np.array([1,2,3,0,2])
        s=paired_labels(r,a,gt,np.arange(5),3,np.array([0.,1.,2.,3.,4.]))
        self.assertEqual(s['mismatches'],5)
        self.assertEqual(s['labeled_count'],4)
        self.assertEqual(s['reference_correct_actual_wrong'],2)
        self.assertEqual(s['reference_wrong_actual_correct'],1)
        self.assertEqual(s['both_wrong_changed'],1)
        self.assertEqual(s['reference_margin_disagreements']['zero_count'],1)
        self.assertIsNone(paired_labels(r,a,gt,[],3,np.ones(5))['actual_OA'])

    def test_22_class_transition_index_does_not_wrap_uint8(self):
        s=paired_labels(np.array([20],dtype=np.uint8),np.array([21],dtype=np.uint8),
                        np.array([21]),np.array([0]),22,np.array([.1]))
        self.assertEqual(s['prediction_transition_matrix'][20][21],1)
        self.assertEqual(np.asarray(s['prediction_transition_matrix']).sum(),1)


class BackendTests(unittest.TestCase):
    def test_frozen_online_algorithms_match_six_hardware_vectors(self):
        anchor=worker.REFERENCE/'hardware_reference'
        if not anchor.is_dir():
            self.skipTest('Historical hardware vectors not included in minimal server package')
        for core in worker.CORES:
            params=json.loads((anchor/f'{core}_parameters.json').read_text())
            cfg=SSMNumericConfig()
            table=compile_coefficients(torch.log(torch.tensor(params['decay_from_checkpoint'],dtype=torch.float64)),
                                       params['s_dt'],params['s_B'],params['s_u'],cfg)
            # Use recorded exact N3 to avoid reconstructing checkpoint theta via log(exp(theta)).
            z=np.load(anchor/f'{core}_n3.npz')
            table.update(a=torch.tensor(z['Abar']),k=torch.tensor(z['K']))
            all_tables=worker.variants(table,params)
            for method in ('n0','n1','n2','n3'):
                z=np.load(anchor/f'{core}_{method}.npz')
                np.testing.assert_array_equal(all_tables[method]['a'],z['Abar'])
                np.testing.assert_array_equal(all_tables[method]['k'],z['K'])

    def test_synthetic_d1_full_tile_all_backends_preserve_n3(self):
        torch.set_num_threads(1)
        torch.manual_seed(193)
        cfg=dict(qat.DEFAULT_MODEL_CONFIG,use_D=True)
        model=qat.build_configured_model(16,9,cfg)
        with contextlib.redirect_stdout(io.StringIO()), torch.no_grad():
            qat.prepare_qat_model(model,model_config=cfg,ssm_contract=dict(u_quantization='shared',d_weight_bits=8))
            model.eval()
            for core in model.modules():
                if isinstance(core,qat.CurrentMambaCore):
                    core.D.copy_(torch.linspace(-.3,.7,core.D.numel()))
            x=torch.rand(1,16,16,16)
            model(x)
            qat.freeze_lsq_initialization(model)
            qat.fuse_qat_model_bns_for_deploy(model,validation_input=x)
            scale=model.patch_embedding[0].lsq_a.s.detach().reshape(())
            expected=worker.base.simulate_tile_dual_int8(copy.deepcopy(model),x,scale,None,torch.device('cpu'))
            with tempfile.TemporaryDirectory() as temp:
                ex=worker.Experiment(trace_tiles=1); ex.out=Path(temp)
                ex.install()
                try:
                    got=worker.base.simulate_tile_dual_int8(copy.deepcopy(model),x,scale,None,torch.device('cpu'))
                    torch.testing.assert_close(expected[2],got[2],rtol=0,atol=0)
                    self.assertEqual(set(ex.tables),set(worker.CORES))
                    self.assertTrue(all(len(v)==1 for v in ex.logits.values()))
                    self.assertEqual(set(ex.saturations),set(ex.methods))
                    for rows in ex.trace['isolated'].values():
                        self.assertTrue(all(v['mismatches']==0 for k,v in rows.items() if '/d_path@' in k))
                    ex.placements=[dict(top=0,left=0,valid_height=16,valid_width=16)]
                    ex.gt=np.ones((16,16),dtype=np.int16)
                    ex.indices=dict(train_indices=np.arange(64),val_indices=np.arange(64,128),test_indices=np.arange(128,256))
                    ex.qat_pred=np.zeros((16,16),dtype=np.uint8)
                    ex.qat_margin=np.ones((16,16))
                    ex.base_pred=ex.float_predictions['n3'][0]
                    ex.finish(dict(dataset='UP',seed=0,qat_checkpoint_sha256='synthetic',numeric_config=SSMNumericConfig().to_dict(),class_count=9))
                    result=json.loads((ex.out/'accuracy_propagation_summary.json').read_text())
                    self.assertEqual(result['status'],'COMPLETE')
                    self.assertFalse(result['hardware_validation_claim'])
                    self.assertEqual(result['methods']['n3']['scopes']['scene']['integer_vs_n3']['mismatches'],0)
                finally:
                    ex.restore()


if __name__=='__main__':
    unittest.main()
