#!/usr/bin/env python3
"""Check file hashes, source syntax and relative documentation links."""
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from urllib.parse import unquote
import xml.etree.ElementTree as ET

from package_backend_error_four_datasets import ROOT, source_files

MANIFEST = ROOT / 'reproducibility_manifest.json'
OWNER_PATH = re.compile(r'~/mzz|/home/music|/Users/zhizhaoma|[DE]:[/\\]|nonlinear_D1_four_datasets_20260927\.zip')
LINK = re.compile(r'!?\[[^\]\n]*\]\(([^)\n]+)\)')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_manifest(files):
    rows = [dict(path=p.relative_to(ROOT).as_posix(), bytes=p.stat().st_size, sha256=sha(p))
            for p in files if p != MANIFEST]
    data = dict(schema=1, revision='third-party-reproducibility-audit-20261008',
                scope='Complete source, docs and small frozen evidence; excludes installed dependencies, '
                      'datasets, checkpoints, build products, caches and this manifest itself.',
                files=rows)
    MANIFEST.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def audit(files, verify_manifest=True):
    result = dict(scope='static_source_identity_and_syntax',
                  files=len(files), bytes=sum(p.stat().st_size for p in files),
                  by_tree=dict(sorted(Counter(p.relative_to(ROOT).parts[0] for p in files).items())),
                  syntax_checks={}, errors=[], hardware_warnings=[], provenance_warnings=[],
                  manifest_checked=verify_manifest)
    counts = Counter()
    for p in files:
        rel = p.relative_to(ROOT).as_posix()
        if p.suffix in {'.py', '.json', '.sh', '.xci', '.md', '.tcl'}:
            text = p.read_text(encoding='utf-8-sig')
        else:
            continue
        try:
            if p.suffix == '.py':
                ast.parse(text, filename=rel); counts['python_ast'] += 1
            elif p.suffix == '.json':
                json.loads(text); counts['json'] += 1
            elif p.suffix == '.xci':
                if text.lstrip().startswith('{'):
                    json.loads(text); counts['xci_json'] += 1
                else:
                    ET.fromstring(text); counts['xci_xml'] += 1
            elif p.suffix == '.sh':
                bash = shutil.which('bash')
                if not bash:
                    raise ValueError('Bash is required for shell syntax checks')
                checked = subprocess.run([bash, '-n', p.as_posix()], capture_output=True,
                                         text=True, encoding='utf-8', errors='replace')
                if checked.returncode:
                    raise ValueError(checked.stderr.strip())
                counts['bash_syntax'] += 1
            elif p.suffix == '.tcl' and shutil.which('tclsh'):
                # info complete parses Tcl quoting without executing Vivado commands.
                checked = subprocess.run([shutil.which('tclsh')], input='set f [open {' + p.name +
                    '} r]\nfconfigure $f -encoding utf-8\nset s [read $f]\nclose $f\nputs [info complete $s]\n',
                    cwd=p.parent, capture_output=True, text=True, encoding='utf-8', errors='replace')
                if checked.returncode or checked.stdout.strip() != '1':
                    raise ValueError('Tcl text is syntactically incomplete: ' + checked.stderr.strip())
                counts['tcl_complete'] += 1
        except (SyntaxError, ValueError, ET.ParseError) as error:
            result['errors'].append(dict(path=rel, check='syntax', detail=str(error)))
        if p.suffix == '.md':
            counts['markdown_links'] += 1
            for match in LINK.finditer(text):
                target = match.group(1).strip().split(' "', 1)[0].strip('<>')
                if target.startswith(('http:', 'https:', 'mailto:', '#', 'data:', 'app:', 'codex:')):
                    continue
                target = unquote(target.split('#', 1)[0])
                if not target:
                    continue
                if not (p.parent / target).exists():
                    item = dict(path=rel, check='relative_link', target=target)
                    result['hardware_warnings' if rel.startswith('hardware/') else 'errors'].append(item)
        for number, line in enumerate(text.splitlines(), 1):
            if OWNER_PATH.search(line) and not (p == Path(__file__).resolve() and line.startswith('OWNER_PATH =')):
                item = dict(path=rel, line=number, check='owner_specific_path')
                if rel.startswith('hardware/'):
                    result['hardware_warnings'].append(item)
                elif rel.startswith(('evidence/', 'third_party/', 'software/snapshots/')) or rel.endswith('_provenance.json'):
                    result['provenance_warnings'].append(item)
                else:
                    result['errors'].append(item)
    result['syntax_checks'] = dict(counts)
    if verify_manifest:
        if not MANIFEST.is_file():
            result['errors'].append(dict(check='manifest', detail='Missing current reproducibility_manifest.json'))
        else:
            expected = json.loads(MANIFEST.read_text(encoding='utf-8'))['files']
            actual = {p.relative_to(ROOT).as_posix(): p for p in files if p != MANIFEST}
            names = {r['path'] for r in expected}
            if len(names) != len(expected):
                result['errors'].append(dict(check='manifest', detail='Duplicate manifest entry'))
            for name in sorted(names - set(actual)):
                result['errors'].append(dict(path=name, check='manifest_missing'))
            for name in sorted(set(actual) - names):
                result['errors'].append(dict(path=name, check='manifest_unlisted'))
            for row in expected:
                p = actual.get(row['path'])
                if p is not None and (p.stat().st_size != row['bytes'] or sha(p) != row['sha256']):
                    result['errors'].append(dict(path=row['path'], check='manifest_hash'))
            result['manifest_entries'] = len(expected)
    result['status'] = 'PASS' if not result['errors'] else 'FAIL'
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-manifest', action='store_true',
                        help='Maintainer operation: replace the current file identity after reviewed source edits')
    parser.add_argument('--json', type=Path, help='Optional report outside the versioned source trees, e.g. results/audit.json')
    args = parser.parse_args(argv)
    files = source_files()
    if args.write_manifest:
        write_manifest(files)
        files = source_files()
    result = audit(files)
    if args.json:
        output = args.json.expanduser().resolve()
        if output == MANIFEST or output in files:
            parser.error('Report must not overwrite source or manifest files')
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    summary = {k: v for k, v in result.items() if k not in {'hardware_warnings', 'provenance_warnings'}}
    for key in ('hardware_warnings', 'provenance_warnings'):
        summary[key + '_count'] = len(result[key])
        summary[key + '_files'] = sorted({r['path'] for r in result[key]})
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
