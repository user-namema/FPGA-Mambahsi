import ast
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from dataclasses import replace

import numpy as np
import torch

import ssm_error_ablation as ab
import train_mambahsi_spatial_split_dense_qat as qat
import both_FPGA_single_qat_source as sim
import analyze_alog_dynamics as analysis


torch.set_num_threads(1)


class NumericalTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(32)
        self.c = ab.SSMNumericConfig()
        self.theta = torch.tensor([0., .5, 1.5])
        self.tables = ab.compile_coefficients(self.theta, .08, .003, .002, self.c)
        self.u = torch.randint(-128, 128, (2, 4, 7))
        self.dt = torch.randint(-128, 128, (2, 7, 4))
        self.b = torch.randint(-128, 128, (2, 3, 7))
        self.readout = torch.randint(-128, 128, (2, 3, 7))

    def oracle(self, config, tables):
        # Independent scalar Python integer implementation, no tensor arithmetic.
        output = torch.empty_like(self.u)
        for batch in range(2):
            h = [[0]*3 for _ in range(4)]
            for t in range(7):
                for channel in range(4):
                    address = int(self.dt[batch, t, channel]) - config.dt_min
                    for n in range(3):
                        pa = int(tables['a'][address,n])*h[channel][n]
                        pu = int(tables['k'][address])*int(self.b[batch,n,t])*int(self.u[batch,channel,t])
                        pu *= 2**(config.a_fraction_bits+config.state_fraction_bits-tables['k_fraction_bits'])
                        def rnd(v):
                            q,r = divmod(abs(v), 2**config.a_fraction_bits)
                            return (-1 if v<0 else 1)*(q+(r>=2**(config.a_fraction_bits-1)))
                        h[channel][n] = max(config.state_min,min(config.state_max,
                            rnd(pa+pu) if config.rounding=='single' else rnd(pa)+rnd(pu)))
                    output[batch,channel,t] = sum(h[channel][n]*int(self.readout[batch,n,t]) for n in range(3))
        return output

    def test_integer_scan_matches_big_integer_oracle(self):
        for c in [self.c, replace(self.c, rounding='separate'),
                  replace(self.c,a_fraction_bits=16,state_bits=28,state_fraction_bits=20),
                  replace(self.c,state_bits=24,state_fraction_bits=24)]:
            tab=ab.compile_coefficients(self.theta,.08,.003,.002,c)
            actual=ab.scan_codes(self.u,self.dt,self.b,self.readout,tab,c)
            self.assertTrue(torch.equal(actual,self.oracle(c,tab)))

    def test_compiler_rejects_unsafe_small_fraction(self):
        with self.assertRaises(OverflowError):
            ab.compile_coefficients(self.theta,1.,100.,20.,self.c)

    def test_65bit_safe_path_uses_python_and_saturates(self):
        a=torch.full((256,3),1<<24,dtype=torch.int64)
        k=torch.full((256,),1<<18,dtype=torch.int64)
        certificate=ab.certify_update_range(a,k,17,self.c)
        self.assertEqual(certificate['required_signed_bits'],65)
        self.assertFalse(certificate['native_int64_safe'])
        h=torch.tensor([[[self.c.state_max]*3]])
        hnew,count=ab.integer_update(h,a[:1].unsqueeze(0),torch.full_like(h,1<<32),17,self.c,False)
        self.assertTrue(torch.all(hnew==self.c.state_max))
        self.assertEqual(count,3)
        with self.assertRaises(OverflowError):
            ab.certify_update_range(a,k,17,replace(self.c,accumulator_bits=64))

    def test_exact_half_ties_and_separate_rounding(self):
        c=replace(self.c,a_fraction_bits=1,state_bits=8,state_fraction_bits=1)
        h=torch.tensor([1,-1]); a=torch.tensor([1,1]); bu=torch.tensor([1,-1])
        single,_=ab.integer_update(h,a,bu,2,c)
        separate,_=ab.integer_update(h,a,bu,2,replace(c,rounding='separate'))
        self.assertEqual(single.tolist(),[1,-1]);self.assertEqual(separate.tolist(),[2,-2])

    def test_nondefault_dt_address_domain(self):
        c=replace(self.c,dt_output_bits=9)
        tab=ab.compile_coefficients(self.theta,.08,.003,.002,c)
        self.assertEqual(tab['a'].shape,(512,3))
        dt=self.dt.clone();dt[0,0,0]=255
        ab.scan_codes(self.u,dt,self.b,self.readout,tab,c)
        with self.assertRaises(ValueError):
            ab.scan_codes(self.u,dt,self.b,self.readout,self.tables,self.c)

    def test_source_isolation_and_zero_reference(self):
        rec=ab.ErrorRecorder()
        c=replace(self.c,error_source='none')
        ab.scan_codes(self.u,self.dt,self.b,self.readout,self.tables,c,recorder=rec)
        self.assertTrue(all(row['squared_error']==0 for row in rec.rows.values()))
        zero=ab.finalize_stats(ab.error_stats(torch.ones(2),torch.zeros(2)))
        self.assertIsNone(zero['relative_l2'])
        for source in ('a-only','k-only','state-only'):
            out=ab.scan_codes(self.u,self.dt,self.b,self.readout,self.tables,replace(self.c,error_source=source))
            self.assertTrue(torch.isfinite(out).all())

    def test_pwl_backend_errors_and_baseline_identity(self):
        config=replace(self.c,coefficient_backend='pwl',pwl_segments=16)
        tables=ab.compile_coefficients(self.theta,.08,.003,.002,config)
        self.assertGreater(tables['coefficient_error']['A_max_abs_error'],0.)
        self.assertLessEqual(tables['coefficient_error']['A_max_abs_error'],1.)
        self.assertFalse(config.rtl_baseline_compatible)
        result=ab.scan_codes(self.u,self.dt,self.b,self.readout,tables,config)
        self.assertTrue(torch.equal(result,self.oracle(config,tables)))

    def test_batch_state_isolation(self):
        together=ab.scan_codes(self.u,self.dt,self.b,self.readout,self.tables)
        parts=[ab.scan_codes(self.u[i:i+1],self.dt[i:i+1],self.b[i:i+1],self.readout[i:i+1],self.tables) for i in range(2)]
        self.assertTrue(torch.equal(together,torch.cat(parts)))

    def test_invalid_config_or_scale_rejected(self):
        for kwargs in [dict(a_fraction_bits=0),dict(state_bits=33),dict(dt_output_bits=1),dict(rounding='bad')]:
            with self.assertRaises(ValueError): ab.SSMNumericConfig(**kwargs)
        with self.assertRaises(ValueError): ab.compile_coefficients(self.theta,0.,.1,.1)
        for poles in ([], [float('nan')], [float('inf')], [1000.], [-1000.]):
            with self.assertRaises(ValueError):
                ab.compile_coefficients(poles, .1, .1, .1)


class IntegrationTests(unittest.TestCase):
    def make_model(self, numeric=None):
        torch.manual_seed(12)
        m=qat.MambaHSI(in_channels=16,hidden_dim=32,num_classes=9)
        qat.prepare_qat_model(m,numeric_config=numeric)
        x=torch.rand(1,16,16,16)
        m.eval()
        with torch.no_grad(): m(x)
        qat.freeze_lsq_initialization(m)
        return m,x

    def test_qat_nondefault_bounds_gradient_and_restore(self):
        c=ab.SSMNumericConfig(dt_input_bits=10,dt_output_bits=7)
        m,x=self.make_model(c)
        y=m(x); y.square().mean().backward()
        core=m.mamba[0].spa_mamba.mamba
        self.assertEqual(core.dt_proj.lsq_a.Qp,511)
        self.assertEqual(core.dt_output_quant.lsq_a.Qp,63)
        self.assertIsNotNone(core.dt_output_quant.lsq_a.s.grad)
        n,_=self.make_model(c);n.load_state_dict(m.state_dict(),strict=True)
        self.assertTrue(torch.equal(m(x),n(x)))
        qat.fuse_qat_model_bns_for_deploy(n,validation_input=x)
        self.assertTrue(qat.validate_fpga_qat_contract(n,expect_bn=False))

    def test_default_full_tile_bit_exact_vs_original_source(self):
        baseline=Path('output/code_backups/20260908_error_ablation/both_FPGA_single_qat_source.py')
        tree=ast.parse(baseline.read_text())
        nodes=[node for node in tree.body if isinstance(node,ast.FunctionDef)]
        namespace=dict(vars(sim))
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(baseline),'exec'),namespace)
        original,x=self.make_model()
        qat.fuse_qat_model_bns_for_deploy(original)
        updated=copy.deepcopy(original)
        scale=original.patch_embedding[0].lsq_a.s.detach().reshape(())
        with torch.no_grad(), contextlib.redirect_stdout(io.StringIO()):
            before=namespace['simulate_tile_dual_int8'](original,x,scale,None,torch.device('cpu'))
            after=sim.simulate_tile_dual_int8(updated,x,scale,None,torch.device('cpu'))
        self.assertTrue(torch.equal(before[2],after[2]),'final integer logits changed')
        self.assertTrue(torch.equal(before[3],after[3]))
        for old, new in zip([m for m in original.modules() if hasattr(m,'A_log_shared')],
                            [m for m in updated.modules() if hasattr(m,'A_log_shared')]):
            for field in ('a_bar_q24','k_multiplier_u19'):
                self.assertTrue(torch.equal(old._fpga_ssm_lut_cache[field],new._fpga_ssm_lut_cache[field]))

    def test_sim_variant_export_and_replay(self):
        c=ab.SSMNumericConfig(dt_input_bits=8,dt_output_bits=9,a_fraction_bits=20,state_bits=28,state_fraction_bits=20)
        m,x=self.make_model(c);qat.fuse_qat_model_bns_for_deploy(m)
        rec=ab.ErrorRecorder()
        for core in [v for v in m.modules() if hasattr(v,'A_log_shared')]:
            core._ssm_error_recorder=rec;core._capture_ssm_inputs=True
        with tempfile.TemporaryDirectory() as folder, torch.no_grad(), contextlib.redirect_stdout(io.StringIO()):
            sim.simulate_tile_dual_int8(m,x,m.patch_embedding[0].lsq_a.s,folder,torch.device('cpu'))
            files=list(Path(folder).glob('ssm_replay_inputs/*.npz'))
            self.assertEqual(len(files),6)
            self.assertGreater(len(rec.rows),0)
            manifest=json.loads((Path(folder)/'ssm_luts/blk0_spa_lut_manifest.json').read_text())
            self.assertFalse(manifest['rtl_baseline_compatible'])
            self.assertEqual(manifest['dt_address_count'],512)

    def test_actual_retention_windows_reset_and_projection(self):
        acc=analysis.RetentionAccumulator(np.array([[1.,2.],[1.,2.]]),[16,24],[1,4,16])
        acc.sequence_shape=(3,4)
        acc.update(torch.full((12,2),-3.))
        rows=acc.rows()
        window=next(r for r in rows if r['kind']=='actual_window_retention' and r['lag']==4)
        self.assertEqual(window['count'],3*2*2)
        self.assertFalse(any(r['lag']==16 for r in rows))
        self.assertTrue(all(r['squared_error']==0 for r in rows if r['kind'].startswith('row_mean')))


if __name__=='__main__': unittest.main()
