import math
import unittest

import torch

import train_mambahsi_spatial_split_dense_qat as qat


class DMaxInitializationTests(unittest.TestCase):
    def test_signed_max_abs_and_supported_widths(self):
        weight=torch.tensor([-.98,1.01,-1.27])
        for bits in (8,12,16):
            q=qat.WeightFakeQuant(bits,init_strategy='max')
            q(weight)
            expected=weight.abs().max()/(2**(bits-1)-1)
            torch.testing.assert_close(q.lsq_w.s[0],expected,rtol=0,atol=0)

    def test_only_d_uses_max_initializer(self):
        config=dict(qat.DEFAULT_MODEL_CONFIG,use_D=True)
        model=qat.build_configured_model(16,9,config)
        qat.prepare_qat_model(model,model_config=config)
        d_count=0
        for name,module in model.named_modules():
            if isinstance(module,qat.LsqQuantizer4weight):
                is_d='.d_weight_quant.' in name
                self.assertEqual(module.init_strategy,'max' if is_d else 'mean')
                d_count+=int(is_d)
        self.assertEqual(d_count,6)
        q=qat.LsqQuantizer4weight(8)
        w=torch.tensor([.9,1.,1.1]);q(w)
        torch.testing.assert_close(q.s[0],2*w.abs().mean()/math.sqrt(127),rtol=0,atol=0)

    def test_calibration_never_reverts_to_mean_and_lsq_stays_trainable(self):
        q=qat.WeightFakeQuant(8,init_strategy='max')
        w=torch.nn.Parameter(torch.tensor([.98,1.013,1.09]))
        for _ in range(100):
            with torch.no_grad():q(w)
        torch.testing.assert_close(q.lsq_w.s[0],w.detach().abs().max()/127,rtol=2e-6,atol=0)
        qat.freeze_lsq_initialization(q)
        q.lsq_w.s.data.fill_(.01)
        q(w).square().sum().backward()
        self.assertIsNotNone(q.lsq_w.s.grad)
        self.assertTrue(torch.isfinite(q.lsq_w.s.grad).all())
        self.assertGreater(float(q.lsq_w.s.grad.abs().max()),0)
        self.assertTrue(torch.isfinite(w.grad).all())
        self.assertAlmostEqual(q.lsq_w.s.item(),.01,places=7)

    def test_zero_d_is_finite(self):
        q=qat.WeightFakeQuant(8,init_strategy='max')
        result=q(torch.zeros(16))
        self.assertTrue(torch.isfinite(result).all())
        self.assertEqual(int(torch.count_nonzero(result)),0)
        self.assertGreater(q.lsq_w.s.item(),0)

    def test_old_checkpoint_saved_scale_is_not_reinitialized(self):
        w=torch.tensor([.95,1.,1.07])
        old=qat.WeightFakeQuant(8)
        old(w);qat.freeze_lsq_initialization(old)
        reference=old(w)
        new=qat.WeightFakeQuant(8,init_strategy='max')
        new.load_state_dict(old.state_dict(),strict=True)
        qat.freeze_lsq_initialization(new)
        self.assertTrue(torch.equal(new(w),reference))
        self.assertTrue(torch.equal(new.lsq_w.s,old.lsq_w.s))


if __name__=='__main__':
    unittest.main()
