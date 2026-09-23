"""Exercise CPU execution when the CUDA extension imports successfully."""
import unittest
from unittest.mock import Mock, patch

import torch
import train_mambahsi_spatial_split_dense_qat as qat
import mambahsi_ablation_model as fp32
import both_FPGA_single_qat_source as sim
from ssm_error_ablation import SSMNumericConfig


class ScanDeviceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(91)

    def test_cpu_scan_bypasses_installed_cuda_extension_with_gradients(self):
        u=torch.randn(2,3,5,requires_grad=True)
        dt=torch.randn(2,3,5,requires_grad=True)
        a=-torch.rand(3,4)
        b=torch.randn(2,4,5)
        c=torch.randn(2,4,5)
        d=torch.randn(3)
        for module in (qat,fp32):
            extra={} if module is qat else {'delta_bias':torch.randn(3)}
            expected=module._reference_selective_scan(u,dt,a,b,c,D=d,**extra)
            with patch.object(module,'_optimized_selective_scan_fn',
                              side_effect=AssertionError('CUDA kernel must not see CPU tensors')) as kernel:
                actual=module.run_selective_scan(u,dt,a,b,c,D=d,**extra)
                kernel.assert_not_called()
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            actual.square().mean().backward()
            self.assertTrue(torch.isfinite(u.grad).all())
            self.assertTrue(torch.isfinite(dt.grad).all())
            u.grad=None;dt.grad=None

    def test_cuda_dispatch_keeps_kernel_and_propagates_its_errors(self):
        # Dispatch-only stand-in: this checks the branch, not GPU numerics.
        u=Mock(is_cuda=True)
        for module in (qat,fp32):
            extra={} if module is qat else {'delta_bias':None}
            marker=object()
            with patch.object(module,'_optimized_selective_scan_fn',return_value=marker) as kernel:
                result=module.run_selective_scan(u,None,None,None,None,D=None,**extra)
                self.assertIs(result,marker)
                self.assertTrue(kernel.call_args.kwargs['delta_softplus'])
                self.assertFalse(kernel.call_args.kwargs['return_last_state'])
            with patch.object(module,'_optimized_selective_scan_fn',side_effect=RuntimeError('kernel failure')):
                with self.assertRaisesRegex(RuntimeError,'kernel failure'):
                    module.run_selective_scan(u,None,None,None,None,D=None,**extra)

    def test_cpu_dt6_qat_reload_and_bn_fusion_with_extension_installed(self):
        config=SSMNumericConfig(dt_input_bits=9,dt_output_bits=6)
        with patch.object(qat,'_optimized_selective_scan_fn',
                          side_effect=AssertionError('Unexpected CUDA dispatch')) as kernel:
            source=qat.MambaHSI(in_channels=16,hidden_dim=32,num_classes=9)
            qat.prepare_qat_model(source,numeric_config=config)
            source.eval()
            x=torch.rand(1,16,16,16)
            with torch.no_grad(): source(x)
            restored=qat.MambaHSI(in_channels=16,hidden_dim=32,num_classes=9)
            qat.prepare_qat_model(restored,numeric_config=config)
            restored.load_state_dict(source.state_dict(),strict=True)
            names,consistency=qat.fuse_qat_model_bns_for_deploy(restored,validation_input=x)
            self.assertTrue(names)
            self.assertTrue(consistency['allclose'])
            qat.validate_fpga_qat_contract(restored,expect_bn=False)
            kernel.assert_not_called()

    def test_simulator_adapter_supports_old_qat_during_bn_fusion(self):
        # Emulate the server bug: old QAT unconditionally calls its CUDA kernel.
        legacy=Mock(side_effect=RuntimeError('Expected u.is_cuda() to be true'))
        config=SSMNumericConfig(dt_input_bits=9,dt_output_bits=6)
        with patch.object(qat,'run_selective_scan',legacy), \
             patch.object(sim,'_qat_run_selective_scan',legacy):
            with sim._device_safe_qat_scan():
                model=qat.MambaHSI(in_channels=16,hidden_dim=32,num_classes=9)
                qat.prepare_qat_model(model,numeric_config=config)
                model.eval()
                x=torch.rand(1,16,16,16)
                with torch.no_grad(): model(x)
                _,check=qat.fuse_qat_model_bns_for_deploy(model,validation_input=x)
                self.assertTrue(check['allclose'])
                u=torch.randn(1,2,3)
                actual=sim.selective_scan_fn(u,u,-torch.ones(2,4),torch.ones(1,4,3),torch.ones(1,4,3))
                self.assertTrue(torch.isfinite(actual).all())
                legacy.assert_not_called()
            self.assertIs(qat.run_selective_scan,legacy)

    def test_simulator_restores_qat_dispatch_after_exception(self):
        previous=qat.run_selective_scan
        with self.assertRaisesRegex(RuntimeError,'test body'):
            with sim._device_safe_qat_scan():
                self.assertIs(qat.run_selective_scan,sim.run_selective_scan)
                raise RuntimeError('test body')
        self.assertIs(qat.run_selective_scan,previous)


if __name__=='__main__': unittest.main()
