import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn
import both_FPGA_single_qat_source as sim

class DTDiagnosticsTests(unittest.TestCase):
    def test_only_actual_dt_slice_and_endpoint_is_not_clipping(self):
        observer=sim.DTInputDiagnostics()
        # Column 1 models B/C values which must not pollute dt statistics.
        acc=torch.tensor([[127,9000],[128,-9000],[-256,9999],[256,-9999]])
        observer.observe('core',1,acc,torch.tensor(1.),torch.tensor(1.),9,acc,acc.float())
        current=observer.rows['core','current'];eight=observer.rows['core','int8_same_scale']
        self.assertEqual(current['count'],4)
        self.assertEqual(current['clipped_count'],1)
        self.assertEqual(current['endpoint_count'],2)
        self.assertEqual(current['raw_code_max'],256)
        self.assertEqual(current['max_abs_error'],1)
        self.assertEqual(eight['clipped_count'],3)
        observer.observe('core',1,acc,torch.tensor(1.),torch.tensor(1.),9,acc,acc.float())
        with tempfile.TemporaryDirectory() as d:
            observer.write(d,'test');r=json.loads((Path(d)/'dt_input_diagnostics.json').read_text())['rows'][0]
            self.assertEqual(r['count'],8);self.assertEqual(r['clipped_pct'],25);self.assertEqual(r['local_MAE'],.25)

    def test_diagnostic_does_not_change_requantized_output(self):
        layer=nn.Linear(2,3,bias=False)
        layer.lsq_w=nn.Module();layer.lsq_w.s=nn.Parameter(torch.tensor(1.))
        layer.nbit_w=8
        with torch.no_grad():layer.weight.copy_(torch.tensor([[1.,2.],[3.,4.],[5.,6.]]))
        x=torch.tensor([[100,50],[-100,-50]],dtype=torch.int8)
        observer=sim.DTInputDiagnostics();ref=torch.nn.functional.linear(x.float(),layer.weight)
        state={k:v.clone() for k,v in layer.state_dict().items()}
        with contextlib.redirect_stdout(io.StringIO()):
            baseline=sim.sim_layer_requant(x,torch.tensor(1.),layer,'core',None,torch.tensor(1.),ref,output_bits=9)
            diagnostic=sim.sim_layer_requant(x,torch.tensor(1.),layer,'core',None,torch.tensor(1.),ref,output_bits=9,dt_input_diagnostic=(observer,'core',1))
        for a,b in zip(baseline,diagnostic):torch.testing.assert_close(a,b,rtol=0,atol=0)
        self.assertTrue(all(torch.equal(v,layer.state_dict()[k]) for k,v in state.items()))
        self.assertFalse(hasattr(layer,'_dt_input_diagnostic'))
        self.assertEqual(observer.rows['core','current']['count'],2)
        self.assertEqual(observer.rows['core','current']['clipped_count'],0)

if __name__=='__main__':unittest.main()
