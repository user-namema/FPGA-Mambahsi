#!/usr/bin/env python3
"""Create a runnable copy of the exact recorded eval1 simulator and its dependencies."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def prepare(output, root=ROOT):
    output = Path(output).resolve()
    software = root / 'software'
    snapshot = software / 'snapshots/eval1_server_20260923'
    hashes = json.loads((snapshot / 'source_hashes.json').read_text())
    selected = {}
    for name, expected in hashes.items():
        source = snapshot / name if (snapshot / name).is_file() else software / name
        actual = hashlib.sha256(source.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError('Source hash mismatch: ' + name)
        selected[name] = source
    if output.exists():
        raise FileExistsError('Choose a new output directory: ' + str(output))
    # No recursive copying of results, snapshots, or caches.
    output.mkdir(parents=True)
    for p in software.glob('*.py'):
        shutil.copy2(p, output / p.name)
    shutil.copytree(software / 'utils', output / 'utils',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.DS_Store'))
    for name, p in selected.items():
        shutil.copy2(p, output / name)
    (output / 'observed_source_hashes.json').write_text(json.dumps(hashes, indent=2) + '\n')
    return output

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    print(prepare(a.output_dir))

if __name__ == '__main__':
    main()
