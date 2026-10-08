"""Small functional test for streaming name repair across chunk boundaries."""
import hashlib
import importlib.util
import json
import tempfile
from pathlib import Path

spec = importlib.util.spec_from_file_location('saif_sanitize', Path(__file__).with_name('saif_sanitize.py'))
saif_sanitize = importlib.util.module_from_spec(spec)
spec.loader.exec_module(saif_sanitize)


def main():
    original = (
        b'(SAIFILE\r\n   (DURATION 327680000)\r\n'
        b'   (NET\r\n      (good\\[0\\] (T0 1) (T1 2) (TX 0) (TZ 0) (TB 0) (TC 1))\r\n'
        b'      (\\\\\\(null\\)\\[0\\]\\.rounded0\\ \\[1\\] '
        b'(T0 1) (T1 2) (TX 0) (TZ 0) (TB 0) (TC 1))\r\n'
        b'   )\r\n)\r\n'
    )
    with tempfile.TemporaryDirectory(prefix='saif_repair_test_') as folder:
        root = Path(folder)
        src, dst = root / 'source.saif', root / 'copy.saif'
        src.write_bytes(original)
        saif_sanitize.CHUNK = 64
        sha = hashlib.sha256(original).hexdigest()
        saif_sanitize.process(src, None, sha, True)
        saif_sanitize.process(src, dst, sha, False)
        copy = dst.read_bytes()
        manifest = json.loads(Path(str(dst) + '.json').read_text(encoding='utf-8'))
        assert src.read_bytes() == original
        assert copy.count(b'XsimNull_') == 1
        assert b'good\\[0\\]' in copy
        assert b'(T0 1) (T1 2) (TX 0) (TZ 0) (TB 0) (TC 1)' in copy
        assert b'null' not in copy
        assert manifest['renamed_names'] == 1
        assert manifest['source_sha256'] == sha
        assert manifest['output_sha256'] == hashlib.sha256(copy).hexdigest()
    print('PASS: source preserved; one NET renamed across chunks; payload, hashes and manifest exact')


if __name__ == '__main__':
    main()
