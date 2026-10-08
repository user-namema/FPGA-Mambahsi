"""Check packaged hardware inputs against their model and simulation contracts."""
import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class DependencyTests(unittest.TestCase):
    def test_full_scene_matches_regression_tiles(self):
        scene = (ROOT / 'models/up_d1/scene/scene_tiles_uint8.bin').read_bytes()
        self.assertEqual(len(scene), 858 * 4096)
        self.assertEqual(hashlib.sha256(scene).hexdigest(),
                         '5e830f01222de08bbffd59d391f73d8130972deee529c75f7506071edf7c752f')
        # Channel zero occupies the least-significant byte of each 128-bit word.
        for name, tile_count in [('models/up_d1/vectors/tiles.mem', 16),
                                 ('board/sim/network_input_tile.mem', 1),
                                 ('board/sim/real5_input_128b.mem', 5)]:
            words = (ROOT / name).read_text(encoding='ascii').split()
            packed = b''.join(int(word, 16).to_bytes(16, 'little') for word in words)
            self.assertEqual(len(packed), tile_count * 4096)
            self.assertEqual(packed, scene[:len(packed)], name)

    def test_real5_logits_match_fullnet_reference(self):
        board = ROOT / 'board/sim'
        real5 = bytes(int(v, 16) for v in
                      (board / 'real5_expected_logits_chw.mem').read_text().split())
        self.assertEqual(len(real5), 5 * 9 * 4 * 4)
        first = bytes(int(v, 16) for v in
                      (board / 'expected_head_logits.mem').read_text().split())
        self.assertEqual(real5[:144], first)
        hwc = bytes(int(v, 16) for v in
                    (ROOT / 'models/up_d1/vectors/n3_gold.mem').read_text().split())
        # Board fixture is tile/channel/y/x; the full-network stream is tile/y/x/channel.
        expected = bytes(hwc[tile * 144 + pixel * 9 + channel]
                         for tile in range(5) for channel in range(9) for pixel in range(16))
        self.assertEqual(real5, expected)

    def test_nonlinear_memories_cover_all_cores_and_methods(self):
        vectors = ROOT / 'nonlinear/vectors'
        manifest = json.loads((vectors / 'manifest.json').read_text(encoding='utf8'))
        self.assertEqual(set(manifest['cores']),
                         {f'blk{block}_{branch}' for block in range(3)
                          for branch in ('spa', 'spe')})
        for core in manifest['cores']:
            with self.subTest(core=core):
                params = json.loads((vectors / (core + '_parameters.json')).read_text(encoding='utf8'))
                self.assertTrue(params)
                for method in range(4):
                    values = (vectors / f'{core}_n{method}.mem').read_text().split()
                    self.assertEqual(len(values), 256)
                    self.assertTrue(all(0 <= int(v, 16) < 2**419 for v in values))
                golden = (vectors / (core + '_d_golden.mem')).read_text().split()
                channels = 64 if core.endswith('_spa') else 16
                self.assertEqual(len(golden), channels * 256)
                self.assertTrue(all(0 <= int(v, 16) < 2**48 for v in golden))

    def test_board_constraints_and_block_design_exist(self):
        profile = json.loads((ROOT / 'config/profiles.json').read_text(encoding='utf8'))['board']
        for rel in [profile['bd'], *profile['constraints']]:
            self.assertGreater((ROOT / rel).stat().st_size, 0, rel)
        bd = json.loads((ROOT / profile['bd']).read_text(encoding='utf8'))
        self.assertIn('design', bd)
        vendor = json.loads((ROOT / 'config/vendor_requirements.json').read_text(encoding='utf8'))
        for item in vendor:
            content = (ROOT / 'vendor/hdmi' / item['name']).read_bytes()
            self.assertEqual(len(content), item['bytes'])
            self.assertEqual(hashlib.sha256(content).hexdigest(), item['sha256'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
