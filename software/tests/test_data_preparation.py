"""Validate preserved dataset values and portable source identity manifests."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from scipy import io as sio


SCRIPT = Path(__file__).resolve().parents[2] / 'tools/prepare_data_formats.py'
SPEC = importlib.util.spec_from_file_location('data_preparation', SCRIPT)
PREP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREP)


class DataPreparationTests(unittest.TestCase):
    def test_npy_conversion_keeps_values_and_manifest_is_portable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root/'HongHu'
            folder.mkdir()
            image = np.arange(3*8*5, dtype=np.float32).reshape(3, 8, 5)
            gt = np.array(list(range(23))+[0], dtype=np.uint8).reshape(3, 8)
            np.save(folder/'WHU_Hi_HongHu.npy', image)
            np.save(folder/'WHU_Hi_HongHu_gt.npy', gt)
            report = root/'reports/data.json'
            command = [sys.executable, str(SCRIPT), '--data-root', str(root),
                       '--datasets', 'HongHu', '--convert-honghu', '--json', str(report)]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            outputs = [folder/'WHU_Hi_HongHu.mat', folder/'WHU_Hi_HongHu_gt.mat']
            np.testing.assert_array_equal(sio.loadmat(outputs[0])['WHU_Hi_HongHu'], image)
            np.testing.assert_array_equal(sio.loadmat(outputs[1])['WHU_Hi_HongHu_gt'], gt)
            details = json.loads(report.read_text())['datasets'][0]
            self.assertEqual(details['data_dtype'], 'float32')
            self.assertEqual(details['label_dtype'], 'uint8')
            for record, path in zip(details['files'], outputs):
                self.assertEqual(record['path'], path.relative_to(root).as_posix())
                self.assertEqual(record['sha256'], PREP.sha256(path))
            previous = [path.read_bytes() for path in outputs]+[report.read_bytes()]
            repeated = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(repeated.returncode, 0)
            self.assertEqual(previous, [path.read_bytes() for path in outputs]+[report.read_bytes()])

    def test_invalid_conversion_leaves_no_mat_or_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root/'HongHu'
            folder.mkdir()
            np.save(folder/'WHU_Hi_HongHu.npy', np.ones((2, 2, 5), dtype=np.float32))
            np.save(folder/'WHU_Hi_HongHu_gt.npy', np.ones((2, 2), dtype=np.uint8))
            with self.assertRaisesRegex(ValueError, 'foreground'):
                PREP.convert_honghu(root, np, sio)
            self.assertFalse(list(folder.glob('*.mat')))
            self.assertFalse((folder/'npy_to_mat_manifest.json').exists())


if __name__ == '__main__':
    unittest.main()
