"""Synthetic checks; no claims about UP accuracy or CUDA reproducibility."""
import copy
import json
import logging
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn

import train_mambahsi_spatial_split_dense_qat as qat
import qat_forensics as forensic
from run_qat_stability_sweep import variants


class ForensicsTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_weight_init_policies_fold_bn_and_restore_scales(self):
        cfg=dict(qat.DEFAULT_MODEL_CONFIG,use_D=True)
        model=qat.build_configured_model(16,9,cfg)
        qat.prepare_qat_model(model,model_config=cfg)
        qat.freeze_lsq_initialization(model)
        with torch.no_grad():
            model.patch_embedding[0].bn_weight.fill_(3.)
        before=copy.deepcopy(model.state_dict())
        for policy,count in [('mean',0),('patch_max',1),('all_weight_max',33)]:
            with forensic.weight_initialization_policy(qat,model,policy) as changes:
                self.assertEqual(len(changes),count)
                if count:
                    layer=model.patch_embedding[0]
                    expected=(layer.weight*3/torch.sqrt(layer.bn_running_var[:,None,None,None]+layer.bn_eps)).abs().max()/127
                    self.assertAlmostEqual(layer.lsq_w.s.item(),expected.item())
                for name,layer in model.named_modules():
                    if isinstance(layer,qat.LsqQuantizer4input) or '.d_weight_quant.lsq_w' in name:
                        self.assertTrue(torch.equal(layer.s,before[name+'.s']))
            self.assertTrue(all(torch.equal(v,model.state_dict()[k]) for k,v in before.items()))
        with self.assertRaisesRegex(RuntimeError,'intentional'):
            with forensic.weight_initialization_policy(qat,model,'all_weight_max'):
                raise RuntimeError('intentional')
        self.assertTrue(all(torch.equal(v,model.state_dict()[k]) for k,v in before.items()))

    def test_all_off_includes_bias_and_restores_on_failure(self):
        model=qat.QuanLinear(2,2,bias=True)
        x=torch.tensor([[.37,.52]])
        model(x);qat.freeze_lsq_initialization(model)
        model.lsq_a.s.data.fill_(.4);model.lsq_w.s.data.fill_(.3)
        original=copy.deepcopy(model.state_dict())
        bias_fn=qat._quantize_accumulator_bias
        with self.assertRaisesRegex(RuntimeError,'intentional'):
            with forensic.quantization_policy(qat,model,[]):
                actual=model(x)
                expected=torch.nn.functional.linear(x,model.weight,model.bias)
                self.assertTrue(torch.equal(actual,expected))
                raise RuntimeError('intentional')
        self.assertIs(qat._quantize_accumulator_bias,bias_fn)
        self.assertTrue(all(torch.equal(v,model.state_dict()[k]) for k,v in original.items()))
        self.assertNotIn('forward',model.lsq_a.__dict__)
        with forensic.quantization_policy(qat,model,forensic.GROUPS):
            actual=model(x)
        self.assertTrue(torch.equal(actual,model(x)))
        self.assertFalse(torch.equal(actual,expected))

    def test_conversion_without_quant_and_group_inventory(self):
        cfg=dict(qat.DEFAULT_MODEL_CONFIG,use_D=True)
        fp=qat.build_configured_model(16,9,cfg).eval()
        model=copy.deepcopy(fp);qat.prepare_qat_model(model,model_config=cfg)
        x=torch.rand(2,16,16,16)
        with torch.no_grad():
            model.eval();model(x);qat.freeze_lsq_initialization(model)
            before=copy.deepcopy(model.state_dict())
            with forensic.quantization_policy(qat,model,[]):
                actual=model(x)
            self.assertTrue(torch.allclose(actual,fp(x),atol=1e-6,rtol=1e-5))
        self.assertTrue(all(torch.equal(v,model.state_dict()[k]) for k,v in before.items()))
        groups=[forensic.quantizer_group(qat,n,m) for n,m in model.named_modules()
                if isinstance(m,(qat.LsqQuantizer4input,qat.LsqQuantizer4weight))]
        self.assertEqual(groups.count('D'),6)
        self.assertEqual(groups.count('dt_input'),6)
        self.assertEqual(groups.count('dt_output'),6)

    def test_execution_context_restores_rng_modes_and_loader_generator(self):
        model=nn.Linear(2,2).train()
        loader=SimpleNamespace(generator=torch.Generator().manual_seed(123))
        state=torch.get_rng_state().clone();gen=loader.generator.get_state().clone()
        with forensic.preserve_execution(model,[loader]):
            torch.rand(5);torch.rand(5,generator=loader.generator);model.eval()
        self.assertTrue(model.training)
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        self.assertTrue(torch.equal(gen,loader.generator.get_state()))

    def test_weight_lr_and_bn_freeze_are_independent_of_D_and_scales(self):
        cfg=dict(qat.DEFAULT_MODEL_CONFIG,use_D=True)
        model=qat.build_configured_model(16,9,cfg);qat.prepare_qat_model(model,model_config=cfg)
        args=SimpleNamespace(lr=1e-5,weight_decay=0.,weight_lr_multiplier=.1,
                             activation_scale_lr_multiplier=1.,dt_scale_lr_multiplier=1.)
        opt=qat.make_qat_optimizer(model,args,logging.getLogger('test'))
        groups={g['name']:g for g in opt.param_groups}
        self.assertAlmostEqual(groups['weights_and_bn']['lr'],1e-6)
        for name in ('bn_affine','d_parameters','activation_scales','weight_scales'):
            self.assertAlmostEqual(groups[name]['lr'],1e-5)
        qat.freeze_bn_affine_parameters(model)
        self.assertTrue(all(not p.requires_grad for p in qat.bn_affine_parameters(model)))
        self.assertTrue(all(p.requires_grad for p in groups['d_parameters']['params']))
        self.assertTrue(all(p.requires_grad for p in groups['weights_and_bn']['params']))
        base=variants()['bn1_fixed_scales']
        self.assertEqual(variants()['fixed_low_weight_lr'],base+['--weight_lr_multiplier','0.1'])
        self.assertEqual(variants()['fixed_freeze_bn_affine'],base+['--freeze_bn_affine'])

    def test_default_optimizer_split_preserves_adam_updates(self):
        model=nn.Sequential(qat.QuanLinear(2,2,norm=True,bias=True))
        original=copy.deepcopy(model)
        args=SimpleNamespace(lr=1e-5,weight_decay=.01,weight_lr_multiplier=1.,
                             activation_scale_lr_multiplier=1.,dt_scale_lr_multiplier=1.)
        opt=qat.make_qat_optimizer(model,args,logging.getLogger('test'))
        ordinary=[p for n,p in original.named_parameters() if not n.endswith('.s')]
        scales=[p for n,p in original.named_parameters() if n.endswith('.s')]
        old=torch.optim.Adam([dict(params=ordinary,lr=args.lr,weight_decay=args.weight_decay),
                              dict(params=scales,lr=args.lr,weight_decay=0.)])
        for step in range(3):
            for p,q in zip(model.parameters(),original.parameters()):
                p.grad=torch.full_like(p,.1*(step+1));q.grad=p.grad.clone()
            opt.step();old.step()
        self.assertTrue(all(torch.equal(p,q) for p,q in zip(model.parameters(),original.parameters())))

    def test_trace_rejects_unfrozen_calibration(self):
        model=nn.Sequential(qat.QuanLinear(2,2))
        with self.assertRaisesRegex(RuntimeError,'Freeze LSQ calibration'):
            forensic.trace_quantized_batch(qat,model,torch.ones(1,1,1,2),2)

    def test_trace_reports_actual_folded_codes_and_restores_hooks(self):
        layer=qat.QuanLinear(2,2,norm=True,bias=True)
        model=nn.Sequential(layer).eval()
        x=torch.tensor([[[[.2,.3]]]])
        # Trace helper expects dense BCHW output: a linear operating on last axis works here.
        model(x);qat.freeze_lsq_initialization(model)
        layer.bn_weight.data.fill_(3.)
        snapshot=copy.deepcopy(model.state_dict())
        a,stats,_=forensic.trace_quantized_batch(qat,model,x,2)
        b,_,_=forensic.trace_quantized_batch(qat,model,x,2)
        self.assertFalse(any(r['changed'] for r in forensic.compare_trace(a,b)))
        folded=layer.weight*layer.bn_weight[:,None]/torch.sqrt(layer.bn_running_var[:,None]+layer.bn_eps)
        expected=qat._LsqRound.apply((folded/layer.lsq_w.s).clamp(layer.lsq_w.Qn,layer.lsq_w.Qp)).to(torch.int32)
        self.assertTrue(torch.equal(a['0.lsq_w/codes#0'][1],expected))
        self.assertIn('threshold_distance_min',stats['0.lsq_w/codes#0'])
        self.assertTrue(all(torch.equal(v,model.state_dict()[k]) for k,v in snapshot.items()))
        self.assertFalse(any(m._forward_hooks for m in model.modules()))

    def test_capture_event_limit_batch_selection_and_offline_replay(self):
        cfg=dict(qat.DEFAULT_MODEL_CONFIG,use_D=True)
        model=qat.build_configured_model(16,9,cfg);qat.prepare_qat_model(model,model_config=cfg)
        model.eval();model(torch.rand(1,16,16,16));qat.freeze_lsq_initialization(model)
        args=SimpleNamespace(model_config=cfg,nbit=8,tile_size=16,eval_batch_size=1,
                             capture_start_epoch=1,capture_oa_drop_pp=5.,capture_max_events=1)
        labels=np.zeros((16,32),dtype=np.int64)
        data=qat.DenseTileDataset(np.ones((16,32,16),np.float32),labels,
            [(0,16,0,16,1),(0,16,16,32,1)],16,False)
        pred=labels.copy();changed=pred.copy();changed[:,16:]=1
        # Artificial metrics force a trigger. The test evaluates capture/replay, not model OA.
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            cap=forensic.JumpCapture(qat,root,args,9,16,qat.SSMNumericConfig(),model._qat_ssm_contract)
            before=copy.deepcopy(model.state_dict())
            cap.observe(model,0,{'OA':1.},pred,data,labels)
            model.patch_embedding[0].weight.data.add_(.01)
            cap.observe(model,1,{'OA':.5},changed,data,labels)
            cap.observe(model,2,{'OA':0.},changed,data,labels)
            self.assertEqual(len(cap.events),1)
            event=Path(cap.events[0]);meta=json.loads((event/'event.json').read_text())
            self.assertEqual(meta['dataset_indices'],[1])
            self.assertTrue(meta['adjacent_epochs'])
            saved=torch.load(event/'before.pth',map_location='cpu')
            self.assertTrue(all(torch.equal(v,saved[k]) for k,v in before.items()))
            proc=subprocess.run([sys.executable,'replay_qat_jump.py','--event-dir',str(event),
                '--output-dir',str(root/'replay'),'--device','cpu'],capture_output=True,text=True)
            self.assertEqual(proc.returncode,0,proc.stdout+proc.stderr)
            report=json.loads((root/'replay/jump_replay.json').read_text())
            self.assertTrue(report['source_hash_match'])
            self.assertTrue(all(x['model_state_unchanged'] for x in report['checkpoints'].values()))
            self.assertTrue(all(x['repeat_logits']['changed']==0 for x in report['checkpoints'].values()))
            self.assertTrue((root/'replay/layer_comparison.csv').exists())


if __name__=='__main__':
    unittest.main()
