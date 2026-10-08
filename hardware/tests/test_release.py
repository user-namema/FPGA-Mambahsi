"""CPU-only packaging checks. These are NOT substitutes for RTL/Vivado tests."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('hw_run',ROOT/'run.py')
run=importlib.util.module_from_spec(spec);spec.loader.exec_module(run)


class ReleaseTests(unittest.TestCase):
    def test_manifest(self):
        self.assertRegex(run.verify(ROOT),r'^[0-9a-f]{64}$')

    def test_profile_files_and_includes(self):
        profiles=json.loads((ROOT/'config/profiles.json').read_text(encoding='utf8'))
        self.assertEqual(set(profiles),{'board','N0','N1','N2','N3'})
        for method,profile in profiles.items():
            names={Path(x).name for x in profile['rtl']}
            self.assertEqual(len(names),len(profile['rtl']),method+' duplicate same-named RTL')
            modules=[]
            for rel in profile['rtl']:
                text=(ROOT/rel).read_text(encoding='utf8',errors='replace')
                modules += re.findall(r'^\s*module\s+(\w+)',text,re.M)
                for name in re.findall(r'`include\s+"([^"]+)"',text):
                    self.assertIn(name,names,method+' missing include')
            self.assertEqual(len(modules),len(set(modules)),method+' duplicate module')

    def test_vectors(self):
        folder=ROOT/'models/up_d1/vectors'
        tiles=(folder/'tiles.mem').read_text().split()
        self.assertEqual(len(tiles),4096)
        self.assertTrue(all(re.fullmatch('[0-9a-fA-F]{32}',s) for s in tiles))
        self.assertEqual(len({tuple(tiles[i*256:(i+1)*256]) for i in range(16)}),16)
        for m in range(4):
            gold=(folder/f'n{m}_gold.mem').read_text().split()
            self.assertEqual(len(gold),2304)
            self.assertTrue(all(re.fullmatch('[0-9a-fA-F]{2}',s) for s in gold))

    def test_ip_initialization(self):
        catalog=json.loads((ROOT/'config/ip_catalog.json').read_text(encoding='utf8'))
        self.assertEqual(len(catalog),85)
        for item in catalog:
            obj=json.loads((ROOT/item['path']).read_text(encoding='utf8'))
            self.assertEqual(obj['ip_inst']['xci_name'],item['name'])
            for old,new in item['relocations'].items():
                self.assertTrue((ROOT/new).is_file(),old)
                self.assertGreater((ROOT/new).stat().st_size,0)

    def test_no_generated_products(self):
        forbidden={'.bit','.dcp','.saif','.wdb','.dll','.exe'}
        files=[p for p in ROOT.rglob('*') if p.is_file()]
        self.assertFalse([p for p in files if p.suffix.lower() in forbidden])
        self.assertLess(max(p.stat().st_size for p in files),100_000_000)
        self.assertFalse(list((ROOT/'rtl').rglob('hdmiout_*.v')))

    def test_tcl_delimiters(self):
        try:
            import tkinter
            interp=tkinter.Tcl()
        except Exception as exc:
            self.skipTest('Tcl/Tk runtime unavailable; run again with a Python including Tcl: '+str(exc))
        for p in ROOT.rglob('*.tcl'):
            self.assertTrue(int(interp.call('info','complete',p.read_text(encoding='utf-8-sig'))),str(p))

    def assert_staged_ips(self, out, expected_count):
        ips = list((out / 'ip').rglob('*.xci'))
        self.assertEqual(len(ips), expected_count)
        for file in ips:
            text = file.read_text(encoding='utf8')
            obj = json.loads(text)
            inst = obj['ip_inst']
            generated = (out / 'generated_ip' / inst['xci_name']).resolve()
            self.assertEqual((file.parent / inst['gen_directory']).resolve(), generated)
            for entry in inst['parameters']['runtime_parameters']['OUTPUTDIR']:
                self.assertEqual((file.parent / entry['value']).resolve(), generated)
            self.assertNotIn('pcie_network_hdmi.gen', text)
            for values in inst['parameters']['component_parameters'].values():
                for item in values:
                    value = item.get('value', '')
                    if isinstance(value, str) and value.lower().endswith('.coe'):
                        self.assertTrue(Path(value).is_file(), value)
                        self.assertIn(out.resolve(), Path(value).resolve().parents)

    def test_portable_stage_and_preserve(self):
        with tempfile.TemporaryDirectory(prefix='hw_release_') as temp:
            for method in ('N0', 'N1', 'N2', 'N3'):
                with self.subTest(profile=method):
                    out = Path(temp) / method
                    with contextlib.redirect_stdout(io.StringIO()):
                        run.stage(method, out, enforce_short_path=False)
                    self.assertEqual(run.verify(out / 'package'), run.verify(ROOT))
                    cfg = (out / 'config.tcl').read_text(encoding='utf8')
                    self.assertIn('set nf_method {' + method + '}', cfg)
                    self.assertNotIn('E:/nf/', cfg)
                    self.assert_staged_ips(out, 83)
                    with self.assertRaisesRegex(RuntimeError, 'NEW'):
                        run.stage(method, out, enforce_short_path=False)

    def test_bundled_vendor_stages_board(self):
        with tempfile.TemporaryDirectory(prefix='hw_board_') as temp:
            out = Path(temp) / 'board'
            with contextlib.redirect_stdout(io.StringIO()):
                run.stage('board', out, enforce_short_path=False)
            requirements = json.loads((ROOT / 'config/vendor_requirements.json').read_text(encoding='utf8'))
            receipt = json.loads((out / 'build_config.json').read_text(encoding='utf8'))
            self.assertEqual(receipt['vendor_files'],
                             [{'name': item['name'], 'sha256': item['sha256']} for item in requirements])
            for item in requirements:
                self.assertEqual(run.digest(out / 'vendor' / item['name']), item['sha256'])
            self.assert_staged_ips(out, 85)
            self.assertTrue((out / 'bd/design_1/design_1.bd').is_file())

    def test_invalid_vendor_override_refused_before_staging(self):
        with tempfile.TemporaryDirectory(prefix='hw_vendor_') as temp:
            vendor = Path(temp) / 'vendor'
            vendor.mkdir()
            required = json.loads((ROOT / 'config/vendor_requirements.json').read_text(encoding='utf8'))
            for kind in ('missing', 'wrong_hash'):
                with self.subTest(kind=kind):
                    if kind == 'wrong_hash':
                        for item in required:
                            (vendor / item['name']).write_text('module invalid; endmodule\n')
                    out = Path(temp) / kind
                    with self.assertRaisesRegex(RuntimeError, 'Vendor source missing or differs'):
                        run.stage('board', out, vendor=vendor, enforce_short_path=False)
                    self.assertFalse(out.exists())


if __name__=='__main__':unittest.main(verbosity=2)
