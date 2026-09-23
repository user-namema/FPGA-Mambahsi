import contextlib
import copy
import io
import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import ssm_error_ablation as ab
from ssm_d_path import compile_d_path, certify_d_requant
import train_mambahsi_spatial_split_dense_qat as qat
import both_FPGA_single_qat_source as sim

torch.set_num_threads(1)


class DArithmeticTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(64)
        self.c=ab.SSMNumericConfig()
        self.tab=ab.compile_coefficients(torch.tensor([0.,1.]),.05,.003,.002,self.c)
        self.u=torch.randint(-128,128,(2,3,7))
        self.dt=torch.randint(-128,128,(2,7,3))
        self.b=torch.randint(-128,128,(2,2,7))
        self.cr=torch.randint(-128,128,(2,2,7))

    def test_signed_channel_bypass_matches_scalar_oracle(self):
        d=compile_d_path(torch.tensor([.75,-1.25,0.]),.002,.007,2,self.c)
        base=ab.scan_codes(self.u,self.dt,self.b,self.cr,self.tab,self.c,scales=(.003,.007))
        actual=ab.scan_codes(self.u,self.dt,self.b,self.cr,self.tab,self.c,scales=(.003,.007),d_tables=d)
        for batch in range(2):
            for ch in range(3):
                for t in range(7):
                    expected=int(base[batch,ch,t])+(int(d['coefficient'][ch])*int(self.u[batch,ch,t])*(2**d['left_shift']))
                    self.assertEqual(int(actual[batch,ch,t]),expected)

    def test_zero_d_is_exact_no_d_and_zero_u_is_zero(self):
        d=compile_d_path(torch.zeros(3),.002,.007,2,self.c)
        baseline=ab.scan_codes(self.u,self.dt,self.b,self.cr,self.tab,self.c,scales=(.003,.007))
        actual=ab.scan_codes(self.u,self.dt,self.b,self.cr,self.tab,self.c,scales=(.003,.007),d_tables=d)
        self.assertTrue(torch.equal(actual,baseline))
        d=compile_d_path(torch.ones(3),.002,.007,2,self.c)
        actual=ab.scan_codes(torch.zeros_like(self.u),self.dt,self.b,self.cr,self.tab,self.c,scales=(.003,.007),d_tables=d)
        self.assertEqual(torch.count_nonzero(actual),0)

    def test_shift_and_bound_certificate(self):
        c=replace(self.c,d_coefficient_bits=8)
        d=compile_d_path(torch.tensor([1.,-2.,0.]),1.,1.,16,c)
        self.assertGreater(d['left_shift'],0)
        self.assertLessEqual(int(d['coefficient'].max()),127)
        self.assertGreaterEqual(int(d['coefficient'].min()),-128)
        with self.assertRaises(OverflowError):
            compile_d_path(torch.tensor([1.]),1.,1.,16,replace(c,d_max_left_shift=0))
        with self.assertRaises(OverflowError):
            compile_d_path(torch.tensor([1.e9]),1.,1.,16,self.c)
        with self.assertRaises(OverflowError):
            certify_d_requant(d,1<<40,0)
        certify_d_requant(d,32767,16)

    def test_signed_half_ties(self):
        d=compile_d_path(torch.tensor([.5,-.5]),2.**-24,1.,2,self.c)
        self.assertEqual(d['coefficient'].tolist(),[1,-1])

    def test_source_reference_and_d_only_is_fold_error(self):
        d=compile_d_path(torch.tensor([.1,-.2,.3]),.002,.007,2,self.c)
        rec=ab.ErrorRecorder()
        ab.scan_codes(self.u,self.dt,self.b,self.cr,self.tab,replace(self.c,error_source='none'),
                      scales=(.003,.007),d_tables=d,recorder=rec)
        self.assertTrue(all(row['squared_error']==0 for row in rec.rows.values()))
        rec=ab.ErrorRecorder()
        ab.scan_codes(self.u,self.dt,self.b,self.cr,self.tab,replace(self.c,error_source='d-only'),
                      scales=(.003,.007),d_tables=d,recorder=rec)
        self.assertTrue(all(v['squared_error']==0 for (name,t),v in rec.rows.items() if name.endswith('/state')))
        self.assertTrue(any(v['squared_error']>0 for (name,t),v in rec.rows.items() if name.endswith('/d_path')))


class DQATTests(unittest.TestCase):
    def make(self,use_d,contract=None):
        torch.manual_seed(45)
        cfg=dict(qat.DEFAULT_MODEL_CONFIG,use_D=use_d)
        m=qat.build_configured_model(16,9,cfg)
        qat.prepare_qat_model(m,model_config=cfg,ssm_contract=contract)
        return m.eval()

    def test_shared_u_single_quantization_and_d_gradients(self):
        m=self.make(True,dict(u_quantization='shared',d_weight_bits=12))
        core=m.mamba[0].spa_mamba.mamba
        x=torch.rand(1,16,16,16)
        with patch.object(core.x_proj.lsq_a,'forward',wraps=core.x_proj.lsq_a.forward) as quant, \
             patch.object(qat,'run_selective_scan',wraps=qat.run_selective_scan) as scan:
            y=m(x)
            self.assertEqual(quant.call_count,1)
            u=scan.call_args_list[0].args[0]
            scale=core.x_proj.lsq_a.s.detach()
            torch.testing.assert_close(u/scale,torch.round(u/scale),rtol=0,atol=1e-5)
        y.square().mean().backward()
        self.assertTrue(torch.isfinite(core.D.grad).all())
        self.assertTrue(torch.isfinite(core.d_weight_quant.lsq_w.s.grad).all())
        self.assertIsNotNone(core.x_proj.lsq_a.s.grad)
        self.assertEqual(core.d_weight_quant.lsq_w.bit,12)
        self.assertEqual(sum(mod.D.numel() for mod in m.modules() if isinstance(mod,qat.CurrentMambaCore)),240)

    def test_zero_d_full_tile_matches_d0_with_shared_u(self):
        contract=dict(u_quantization='shared',d_weight_bits=8)
        m0=self.make(False,contract); x=torch.rand(1,16,16,16)
        with torch.no_grad(): m0(x)
        qat.freeze_lsq_initialization(m0)
        m1=self.make(True,contract)
        mismatch=m1.load_state_dict(m0.state_dict(),strict=False)
        self.assertFalse(mismatch.unexpected_keys)
        self.assertTrue(all('d_weight_quant' in k for k in mismatch.missing_keys))
        qat.freeze_lsq_initialization(m1)
        with torch.no_grad():
            torch.testing.assert_close(m0(x),m1(x),rtol=0,atol=0)
        for m in (m0,m1): qat.fuse_qat_model_bns_for_deploy(m,validation_input=x)
        with torch.no_grad(),contextlib.redirect_stdout(io.StringIO()):
            a=sim.simulate_tile_dual_int8(m0,x,m0.patch_embedding[0].lsq_a.s.detach().reshape(()),None,torch.device('cpu'))
            b=sim.simulate_tile_dual_int8(m1,x,m1.patch_embedding[0].lsq_a.s.detach().reshape(()),None,torch.device('cpu'))
        self.assertTrue(torch.equal(a[2],b[2]))

    def test_nonzero_d_reload_fuse_and_export(self):
        contract=dict(u_quantization='shared',d_weight_bits=8)
        m=self.make(True,contract);x=torch.rand(1,16,16,16)
        with torch.no_grad(): m(x)
        qat.freeze_lsq_initialization(m)
        restored=self.make(True,contract);restored.load_state_dict(m.state_dict(),strict=True)
        qat.freeze_lsq_initialization(restored)
        with torch.no_grad(): torch.testing.assert_close(m(x),restored(x),rtol=0,atol=0)
        qat.fuse_qat_model_bns_for_deploy(restored,validation_input=x)
        with tempfile.TemporaryDirectory() as folder,torch.no_grad(),contextlib.redirect_stdout(io.StringIO()):
            sim.simulate_tile_dual_int8(restored,x,restored.patch_embedding[0].lsq_a.s.detach().reshape(()),folder,torch.device('cpu'))
            manifests=list(Path(folder).glob('*_d_manifest.json'))
            self.assertEqual(len(manifests),6)
            self.assertEqual(sum(json.loads(p.read_text())['coefficient_count'] for p in manifests),240)
            self.assertEqual(len(list(Path(folder).glob('*_ssm_y_d_codes.txt'))),6)


if __name__=='__main__': unittest.main()
