#!/usr/bin/env python3
"""Read-only backup of installed scan/causal extensions; no imports of Torch/CUDA."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import platform
import sys
import sysconfig
import tarfile
import tempfile

EXPECTED = {
    'selective_scan_cuda': '25e78f53d6aafbb7ac11be80d720fd07200b048ac3f7800778248c74dd99bd45',
    'causal_conv1d_cuda': '8cefee9d05467dc5a288ab2e2e3789ffc46af9c7bed939b0386c9da34bef3d83',
}
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

def backup(site, output):
    site = Path(site).resolve()
    files = []
    for module in EXPECTED:
        candidates = sorted(site.glob(module + '*.so'))
        if len(candidates) != 1:
            raise RuntimeError('Expected one installed ' + module + ' extension; found ' + str(len(candidates)))
        files.extend(candidates)
    package = site / 'causal_conv1d'
    if not package.is_dir(): raise FileNotFoundError(str(package))
    files.extend(sorted(package.rglob('*.py')))
    for pattern in ('causal_conv1d-*.dist-info', 'mamba_ssm-*.dist-info'):
        for directory in site.glob(pattern):
            files.extend(p for p in directory.iterdir() if p.is_file())
    rows = []
    for p in files:
        if p.is_symlink() or site not in p.resolve().parents:
            raise ValueError('Refusing a symlink or external path: ' + str(p))
        digest = sha(p)
        row = dict(path=str(p.relative_to(site)), bytes=p.stat().st_size, sha256=digest)
        for prefix, expected in EXPECTED.items():
            if p.name.startswith(prefix) and p.suffix == '.so':
                row.update(recorded_20260923_sha256=expected, matches_recorded_20260923=(digest == expected))
        rows.append(row)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    destination = Path(tempfile.mkdtemp(prefix='cuda_runtime_', dir=str(output)))
    manifest = dict(python=sys.version, platform=platform.platform(), files=rows,
                    note='Installed runtime backup, not a wheel or cross-platform installer.')
    data = (json.dumps(manifest, indent=2) + '\n').encode()
    archive = destination / 'installed_cuda_runtime.tar.gz'
    with tarfile.open(str(archive), 'w:gz') as tar:
        for p in files: tar.add(str(p), arcname='site-packages/' + str(p.relative_to(site)), recursive=False)
        info = tarfile.TarInfo('runtime_manifest.json'); info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    (destination / 'runtime_manifest.json').write_bytes(data)
    (destination / 'installed_cuda_runtime.tar.gz.sha256').write_text(sha(archive) + '  ' + archive.name + '\n')
    return archive

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, default=Path('runtime_backup'))
    args = p.parse_args()
    print(backup(Path(sysconfig.get_paths()['platlib']), args.output_dir))

if __name__ == '__main__': main()
