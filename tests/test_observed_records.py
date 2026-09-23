import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'tools' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
class ObservedRecordTests(unittest.TestCase):
    def test_bundled_evidence_is_consistent(self):
        result = load('verify_release_records').verify(ROOT)
        self.assertEqual(result['fpga_eval1_runs'], 40)
        self.assertEqual(result['power_trials'], 42)
    def test_prepare_exact_source_and_refuse_existing_output(self):
        prepare = load('prepare_observed_eval1').prepare
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'source'
            prepare(target, ROOT)
            snapshot = ROOT / 'software/snapshots/eval1_server_20260923'
            for name in ['both_FPGA_single_qat_source.py', 'run_fpga_qat_eval1_four_datasets.py']:
                self.assertEqual((target / name).read_bytes(), (snapshot / name).read_bytes())
            self.assertTrue((target / 'utils/evaluation.py').is_file())
            self.assertFalse((target / 'snapshots').exists())
            with self.assertRaises(FileExistsError): prepare(target, ROOT)
    def test_bad_dependency_hash_fails_before_output_creation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            snapshot = root / 'software/snapshots/eval1_server_20260923'
            snapshot.mkdir(parents=True)
            (root / 'software/dependency.py').write_text('changed')
            (snapshot / 'source_hashes.json').write_text(json.dumps({'dependency.py': '0'*64}))
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                load('prepare_observed_eval1').prepare(root / 'out', root)
            self.assertFalse((root / 'out').exists())
if __name__ == '__main__': unittest.main()
