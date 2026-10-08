"""Verify frozen hardware release; print its manifest SHA256 for Tcl callers."""
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(root):
    root = Path(root).resolve()
    manifest = root / 'manifest.json'
    data = json.loads(manifest.read_text(encoding='utf-8'))
    for entry in data['files']:
        path = (root / entry['path']).resolve()
        if root not in path.parents or not path.is_file():
            raise RuntimeError('Missing/unsafe hardware file: ' + entry['path'])
        if path.stat().st_size != entry['bytes'] or digest(path) != entry['sha256']:
            raise RuntimeError('Hardware file changed: ' + entry['path'])
    return digest(manifest)


if __name__ == '__main__':
    print(verify(Path(__file__).resolve().parents[1]))
