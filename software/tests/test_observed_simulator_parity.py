"""Check the diagnostic-enabled simulator against the received server revision."""
import contextlib
import copy
import importlib.util
import io
from pathlib import Path
import sys
import unittest
import torch
import train_mambahsi_spatial_split_dense_qat as qat
import both_FPGA_single_qat_source as current
class RecordedSimulatorParityTests(unittest.TestCase):
    def test_nonzero_d_full_tile_matches_recorded_revision(self):
        path = Path(__file__).resolve().parents[1] / 'snapshots/eval1_server_20260923/both_FPGA_single_qat_source.py'
        spec = importlib.util.spec_from_file_location('recorded_eval1_sim', path)
        recorded = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = recorded
        spec.loader.exec_module(recorded)
        torch.manual_seed(193)
        cfg = dict(qat.DEFAULT_MODEL_CONFIG, use_D=True)
        model = qat.build_configured_model(16, 9, cfg)
        qat.prepare_qat_model(model, model_config=cfg, ssm_contract=dict(u_quantization='shared', d_weight_bits=8))
        model.eval()
        with torch.no_grad():
            for core in model.modules():
                if isinstance(core, qat.CurrentMambaCore):
                    core.D.copy_(torch.linspace(-0.3, 0.7, core.D.numel()).reshape_as(core.D))
            x = torch.rand(1, 16, 16, 16)
            model(x)
            qat.freeze_lsq_initialization(model)
            qat.fuse_qat_model_bns_for_deploy(model, validation_input=x)
            with contextlib.redirect_stdout(io.StringIO()):
                outputs = []
                for backend in (recorded, current):
                    m = copy.deepcopy(model)
                    outputs.append(backend.simulate_tile_dual_int8(m, x, m.patch_embedding[0].lsq_a.s.detach().reshape(()), None, torch.device('cpu')))
        torch.testing.assert_close(outputs[0][2], outputs[1][2], rtol=0, atol=0)
        torch.testing.assert_close(outputs[0][3], outputs[1][3], rtol=0, atol=0)
if __name__ == '__main__': unittest.main()
