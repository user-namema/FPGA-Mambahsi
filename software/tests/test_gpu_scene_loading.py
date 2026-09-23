"""Exercise the current three-value loader contract through saved preprocessing."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import benchmark_gpu_batch1_dense16 as benchmark


class SceneLoadingTest(unittest.TestCase):
    def test_current_loader_and_partial_edge_tiles(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            shape = (17, 19)
            raw = np.ones(shape + (16,), dtype=np.float32)
            gt = np.ones(shape, dtype=np.int64)
            np.savez(root/'train_only_preprocess.npz', pca_mean=np.zeros(16),
                     pca_components=np.eye(16), explained_variance=np.ones(16),
                     channel_min=np.zeros(16), channel_max=np.ones(16))
            np.savez(root/'spatial_split_masks.npz', train_region=np.ones(shape, bool),
                     val_region=np.zeros(shape, bool), test_region=np.zeros(shape, bool))
            blocks = [dict(top=t, bottom=min(t+16,17), left=l, right=min(l+16,19), split='train')
                      for t in (0,16) for l in (0,16)]
            (root/'spatial_split.json').write_text(json.dumps(dict(strategy='blocks',
                block_size=16, effective_tile_size=16, blocks=blocks)))
            with patch.object(benchmark, 'load_dataset', return_value=(raw, gt, 1)):
                image, labels, classes, actual, elapsed = benchmark.load_scene_and_blocks(
                    SimpleNamespace(dataset='synthetic', data_path='unused'), root)
            self.assertEqual(image.shape, (17,19,16))
            np.testing.assert_array_equal(labels, gt)
            self.assertEqual(classes, 1)
            self.assertEqual(len(actual), 4)
            self.assertGreaterEqual(elapsed, 0)


if __name__ == '__main__': unittest.main()
