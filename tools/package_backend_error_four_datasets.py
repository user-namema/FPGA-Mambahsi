#!/usr/bin/env python3
"""Optionally bundle the complete reproduction source, docs and small evidence.

No dataset, trained checkpoint, installed runtime or generated FPGA build is
included. Experiments can be run directly from a checkout; a ZIP is optional.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
TREES = ('software', 'experiments', 'environment', 'tools', 'tests', 'docs',
         'evidence', 'third_party', 'hardware')
EXCLUDED = {'.git', '__pycache__', '.pytest_cache', '.DS_Store', 'vendor_local',
            '.Xil', '.venv'}
GENERATED = {'.pyc', '.pyo', '.bit', '.dcp', '.wdb', '.saif', '.jou', '.log',
             '.pth', '.pt', '.ckpt', '.so', '.dll', '.dylib'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_files():
    files = [p for p in ROOT.iterdir() if p.is_file() and
             p.name in {'.gitignore', 'README.md', 'README_zh-CN.md',
                        'source_manifest.json', 'release_manifest.json',
                        'reproducibility_manifest.json', 'LICENSE'}]
    for tree in TREES:
        for p in (ROOT / tree).rglob('*'):
            if (not p.is_file() or p.is_symlink() or
                any(x in EXCLUDED or x.startswith('build') or
                    x.endswith(('.runs', '.cache', '.gen', '.ip_user_files'))
                    for x in p.relative_to(ROOT).parts[:-1]) or
                p.name in EXCLUDED or p.suffix in GENERATED):
                continue
            # Bundled .npy/.npz files are small frozen evidence, not raw scenes.
            if p.suffix in {'.mat', '.npy', '.npz'} and tree not in {'evidence', 'software'}:
                continue
            files.append(p)
    return sorted(files)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--refresh', action='store_true',
                        help='Refresh only an unchanged bundle previously generated here')
    args = parser.parse_args(argv)
    target = args.output_dir.expanduser().resolve()
    archive = Path(str(target) + '.zip')
    if target == ROOT or target in ROOT.parents:
        parser.error('The bundle directory must not contain the source repository')
    if args.refresh:
        old = json.loads((target / 'SHA256SUMS.json').read_text(encoding='utf-8'))
        actual = {p.relative_to(target).as_posix() for p in target.rglob('*')
                  if p.is_file() and p.name != 'SHA256SUMS.json'}
        if actual != set(old):
            raise ValueError('Bundle has missing or extra files; refusing refresh')
        for name, expected in old.items():
            p = target / name
            if p.is_symlink() or target not in p.resolve().parents or digest(p) != expected:
                raise ValueError('Bundle modified; refusing refresh: ' + name)
    elif target.exists() or archive.exists():
        raise FileExistsError('Use a new bundle path; existing files are preserved')
    sources = source_files()
    if any(target in p.parents for p in sources):
        parser.error('Choose an output directory outside the source trees')
    for source in sources:
        dest = target / source.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
    # Retain the optional legacy launcher at the correct location.
    launcher = target / 'run.sh'
    launcher.write_text('#!/usr/bin/env bash\nset -euo pipefail\n'
                        'SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"\n'
                        'exec "${PYTHON_BIN:-python}" '
                        '"$SCRIPT_DIR/software/run_backend_error_four_datasets.py" "$@"\n',
                        encoding='utf-8')
    manifest = {p.relative_to(target).as_posix(): digest(p)
                for p in sorted(target.rglob('*'))
                if p.is_file() and p.name != 'SHA256SUMS.json'}
    (target / 'SHA256SUMS.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    with zipfile.ZipFile(archive, 'w' if args.refresh else 'x',
                         compression=zipfile.ZIP_DEFLATED) as z:
        for name in [*manifest, 'SHA256SUMS.json']:
            p = target / name
            z.write(p, p.relative_to(target.parent))
    print(json.dumps(dict(directory=str(target), archive=str(archive),
                          files=len(manifest) + 1, bytes=archive.stat().st_size,
                          sha256=digest(archive)), indent=2))


if __name__ == '__main__':
    main()
