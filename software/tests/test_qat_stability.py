import copy
import logging
from types import SimpleNamespace
import unittest

import torch
from torch import nn
import train_mambahsi_spatial_split_dense_qat as qat
from run_qat_stability_sweep import variants


class StabilityTests(unittest.TestCase):
    def test_freezing_clears_stale_grad_and_adam_momentum_updates(self):
        model=nn.Sequential(qat.QuanLinear(2,2,bias=False))
        x=torch.tensor([[.43,.74]])
        model(x);qat.freeze_lsq_initialization(model)
        opt=torch.optim.Adam(model.parameters(),lr=.01)
        model(x).sum().backward();opt.step()
        qat.freeze_learned_lsq_scales(model)
        before=[m.s.detach().clone() for m in model.modules() if isinstance(m,(qat.LsqQuantizer4input,qat.LsqQuantizer4weight))]
        opt.zero_grad(set_to_none=True);model(x).sum().backward();opt.step()
        after=[m.s for m in model.modules() if isinstance(m,(qat.LsqQuantizer4input,qat.LsqQuantizer4weight))]
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(before,after)))
        self.assertTrue(model[0].weight.requires_grad)

    def test_probe_reads_clipping_and_restores_modes_and_state(self):
        model=nn.Sequential(qat.QuanLinear(2,2,bias=False))
        model(torch.ones(1,2));qat.freeze_lsq_initialization(model)
        model[0].lsq_a.s.data.fill_(.01)
        model.train();snapshot=copy.deepcopy(model.state_dict())
        state=torch.get_rng_state().clone()
        rows=qat.fixed_batch_quantization_probe(model,torch.tensor([[10.,0.]]))
        self.assertAlmostEqual(rows['0.lsq_a']['clipping_ratio'],.5)
        self.assertAlmostEqual(rows['0.lsq_a']['zero_ratio'],.5)
        self.assertTrue(model.training and model[0].training)
        self.assertTrue(all(torch.equal(v,model.state_dict()[k]) for k,v in snapshot.items()))
        self.assertTrue(torch.equal(state,torch.get_rng_state()))

    def test_cosine_and_constant_rates(self):
        p=nn.Parameter(torch.ones(1));opt=torch.optim.Adam([p],lr=.001)
        qat.set_qat_epoch_lr(opt,1,10,'cosine',.1);self.assertAlmostEqual(opt.param_groups[0]['lr'],.001)
        qat.set_qat_epoch_lr(opt,10,10,'cosine',.1);self.assertAlmostEqual(opt.param_groups[0]['lr'],.0001)
        qat.set_qat_epoch_lr(opt,10,10,'constant',.1);self.assertAlmostEqual(opt.param_groups[0]['lr'],.001)

    def test_d_scale_group_and_nonduplicated_drift(self):
        cfg=dict(qat.DEFAULT_MODEL_CONFIG,use_D=True)
        m=qat.build_configured_model(16,9,cfg);qat.prepare_qat_model(m,model_config=cfg)
        args=SimpleNamespace(lr=1e-5,dt_scale_lr_multiplier=.2,activation_scale_lr_multiplier=.3,
            weight_scale_lr_multiplier=.4,d_scale_lr_multiplier=.5,weight_decay=0.)
        opt=qat.make_qat_optimizer(m,args,logging.getLogger('test'))
        group=next(g for g in opt.param_groups if g['name']=='d_weight_scales')
        self.assertEqual(len(group['params']),6);self.assertAlmostEqual(group['lr'],5e-6)
        _,before=qat.qat_drift_diagnostics(m)
        group['params'][0].data.mul_(1.1)
        report,_=qat.qat_drift_diagnostics(m,before)
        changed=[r for r in report['scales'] if abs(r['relative_change'])>.01]
        self.assertEqual(len(changed),1)
        self.assertAlmostEqual(changed[0]['relative_change'],.1,places=6)

    def test_paired_runner_primary_arms(self):
        v=variants()
        self.assertEqual(v['bn1_fixed_scales'],v['bn1']+['--freeze_lsq_epoch','1'])
        self.assertEqual(v['control_bn20'],['--freeze_bn_epoch','20'])


if __name__=='__main__':unittest.main()
