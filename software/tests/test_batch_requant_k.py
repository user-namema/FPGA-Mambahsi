import unittest
from dataclasses import replace
from unittest.mock import patch

import torch
import numpy as np
from torch import nn

import train_mambahsi_spatial_split_dense_qat as qat
import both_FPGA_single_qat_source as sim
from ssm_error_ablation import SSMNumericConfig, compile_coefficients, integer_update
from run_ssm_error_ablation import configurations
from diagnose_qat_batch import trace_tile, compare_traces, first_differences, backend_mode, select_mismatch_group


class ReadoutTests(unittest.TestCase):
    def test_ideal_integer_readout_signed_ties_and_saturation(self):
        c=replace(SSMNumericConfig(),readout_requantization='ideal')
        values=torch.tensor([-1000,-5,-3,-1,0,1,3,5,1000],dtype=torch.int64)
        actual=sim.ssm_readout_to_int8(values,torch.tensor(.25),torch.tensor(.5),c)
        self.assertEqual(actual.tolist(),[-128,-3,-2,-1,0,1,2,3,127])

    def test_hardware_default_preserves_existing_requantizer(self):
        c=SSMNumericConfig()
        x=torch.arange(-10000,10000,31,dtype=torch.int64)
        s=torch.tensor(.00317);o=torch.tensor(.119)
        expected=sim.requantize_int8(x,s,o)
        self.assertTrue(torch.equal(sim.ssm_readout_to_int8(x,s,o,c),expected))
        self.assertTrue(torch.equal(sim.ssm_readout_to_int8(x,s,o,replace(c,readout_requantization='hardware')),expected))

    def test_counterfactual_already_physical_and_invalid_hardware(self):
        c=replace(SSMNumericConfig(),error_source='none')
        out=sim.ssm_readout_to_int8(torch.tensor([-.75,.75]),torch.tensor(.0001),torch.tensor(.5),c)
        self.assertEqual(out.tolist(),[-2,2])
        with self.assertRaises(ValueError):
            replace(c,readout_requantization='hardware')

    def test_requant_pair_only_changes_output_boundary(self):
        cases=dict(configurations(SSMNumericConfig(),'requant'))
        ideal=cases['all_ideal'].to_dict();hw=cases['all_hardware'].to_dict()
        self.assertEqual([k for k in ideal if ideal[k]!=hw[k]],['readout_requantization'])
        h=torch.tensor([1,-1]);a=torch.tensor([5,7]);bu=torch.tensor([3,-4])
        self.assertTrue(torch.equal(integer_update(h,a,bu,24,cases['all_ideal'])[0],
                                    integer_update(h,a,bu,24,cases['all_hardware'])[0]))


class KGridTests(unittest.TestCase):
    def test_grid_compiles_actual_fraction_and_reduces_coefficient_error(self):
        c=SSMNumericConfig()
        cases=dict(configurations(c,'k-precision'))
        self.assertEqual(len(cases),9)
        tables={name:compile_coefficients(torch.tensor([0.,1.]),.05,.003,.002,cfg) for name,cfg in cases.items()}
        low=tables['K_b19_fmax24'];high=tables['K_b23_fmax28']
        self.assertGreater(high['k_fraction_bits'],low['k_fraction_bits'])
        self.assertLess(high['coefficient_error']['K_max_abs_error'],low['coefficient_error']['K_max_abs_error'])
        self.assertEqual(high['logical_rom_bits']-low['logical_rom_bits'],256*4)
        self.assertEqual(len(configurations(c,'k-precision',k_bits_grid=[19,19],k_fraction_grid=[24,26])),2)
        with self.assertRaises(ValueError):
            configurations(c,'k-precision',k_bits_grid=[40])


class BatchTraceTests(unittest.TestCase):
    def test_auto_selection_uses_only_labeled_test_pixels(self):
        one=np.zeros((16,64),dtype=np.int64);many=one.copy()
        many[0,0]=1  # background mismatch must not select the first tile
        many[0,49]=1
        blocks=[dict(top=0,bottom=16,left=i*16,right=(i+1)*16) for i in range(4)]
        self.assertEqual(select_mismatch_group(one,many,np.array([49]),blocks,2),(1,1))
        with self.assertRaises(ValueError):
            select_mismatch_group(one,many,np.array([50]),blocks,2)

    def make_model(self,mix=False):
        class Toy(nn.Module):
            def __init__(self):
                super().__init__()
                self.quant=qat.LsqQuantizer4input(8)
                self.quant.init_state=self.quant.batch_init
                self.quant.s.data.fill_(.1)
                self.act=nn.ReLU()
            def forward(self,x):
                if mix:
                    x=x+x.mean(0,keepdim=True)
                return self.act(self.quant(x)[0])
        return Toy().eval()

    def test_target_tile_mapping_and_code_divergence(self):
        x=torch.stack([torch.ones(3,16,16),torch.ones(3,16,16)*2])
        m=self.make_model()
        single,_=trace_tile(m,x[1:2]);batch,_=trace_tile(m,x,target=1)
        self.assertIsNone(first_differences(compare_traces(single,batch,'independent'))['first_exact'])
        m=self.make_model(mix=True)
        single,_=trace_tile(m,x[1:2]);batch,_=trace_tile(m,x,target=1)
        first=first_differences(compare_traces(single,batch,'mixed'))
        self.assertEqual(first['first_code_change']['trace'],'quant/codes#0')
        self.assertEqual(first['first_above_tolerance']['trace'],'quant/input#0')

    def test_restore_hooks_and_backend_after_failure(self):
        m=self.make_model();original=qat.run_selective_scan
        tf32=torch.backends.cuda.matmul.allow_tf32
        with self.assertRaises(RuntimeError):
            with backend_mode('reference-tf32-off'):
                with patch.object(m,'forward',side_effect=RuntimeError('probe failure')):
                    trace_tile(m,torch.ones(2,3,16,16))
        self.assertIs(qat.run_selective_scan,original)
        self.assertEqual(torch.backends.cuda.matmul.allow_tf32,tf32)
        self.assertTrue(all(not module._forward_hooks and not module._forward_pre_hooks for module in m.modules()))


if __name__=='__main__':
    unittest.main()
