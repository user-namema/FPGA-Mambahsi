"""N3-only post-route SAIF capture/report runner. No synthesis, route or deletion."""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def config_value(text, name):
    m = re.search(r'^set ' + re.escape(name) + r' ([^\r\n]+)', text, re.M)
    if not m:
        raise RuntimeError('Missing config field: ' + name)
    return m[1].strip().strip('{}')


def preflight(build, mode, min_free):
    for rel in ('nf_fullnet.xpr', 'config.tcl', 'sim_ok.txt', 'build_status.txt',
                'network_routed.dcp', 'package/scripts/sim.tcl', 'package/tools/verify.py'):
        if not (build / rel).is_file():
            raise RuntimeError('Missing required file: ' + str(build / rel))
    cfg = (build / 'config.tcl').read_text(encoding='utf-8')
    if config_value(cfg, 'nf_method') != 'N3' or float(config_value(cfg, 'nf_period_ns')) != 10:
        raise RuntimeError('This entry requires the routed N3 100-MHz build')
    expected = config_value(cfg, 'nf_source_hash')
    actual = subprocess.check_output([sys.executable, '-I', str(build / 'package/tools/verify.py')], text=True).strip()
    if expected != actual:
        raise RuntimeError('Frozen package hash mismatch; original files will not be changed')
    if (build / 'sim_ok.txt').read_text(encoding='utf-8').splitlines()[0] != expected:
        raise RuntimeError('The 16-tile RTL PASS belongs to another package')
    if (build / 'build_status.txt').read_text(encoding='utf-8').splitlines()[0] != 'ROUTED':
        raise RuntimeError('No completed routed result')
    free_gib = shutil.disk_usage(build).free / 2**30
    print(f'BUILD: {build}\nFREE: {free_gib:.2f} GiB\nSOURCE: {actual}', flush=True)
    if mode != 'check' and free_gib < min_free:
        raise RuntimeError(f'Need at least {min_free:g} GiB free for this run; no automatic cleanup is performed')
    return {'build': str(build), 'source_sha256': actual,
            'routed_dcp_sha256': digest(build / 'network_routed.dcp'), 'free_gib': free_gib}


def invoke(vivado, mode, build, out, capture, report_saif=None, report_saif_hash=None):
    script = Path(__file__).resolve().parent / 'n3_power.tcl'
    args = [str(vivado), '-mode', 'batch', '-nojournal', '-nolog', '-source', str(script),
            '-tclargs', mode, build.as_posix(), out.as_posix(), capture.as_posix(), sys.executable]
    if report_saif is not None:
        args.extend((report_saif.as_posix(), report_saif_hash))
    log = out / (mode + '_console.log')
    print(f'{mode.upper()} CONSOLE: {log}', flush=True)
    with log.open('x', encoding='utf-8') as f:
        proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for raw in iter(proc.stdout.readline, b''):
            line = raw.decode('utf-8', errors='replace')
            f.write(line)
            f.flush()
            print(line, end='', flush=True)
        code = proc.wait()
    if code:
        raise RuntimeError(f'{mode} failed, exit={code}; inspect {log}. Existing captures/reports retained.')


def runner_command(mode, build, *args, vivado=None):
    command = [sys.executable, '-I', str(Path(__file__).resolve()),
               mode, '--build', str(build), *map(str, args)]
    if vivado is not None:
        command.extend(('--vivado', str(vivado)))
    return subprocess.list2cmdline(command)


def main():
    # Tcl calls this mode for streaming hashes without loading Vivado's Python libs.
    if len(sys.argv) == 3 and sys.argv[1] == '--sha256':
        print(digest(sys.argv[2]))
        return
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', nargs='?', choices=('check', 'probe', 'all', 'capture', 'report'), default='probe')
    p.add_argument('--build', required=True, help='Completed portable N3 build')
    p.add_argument('--out', help='NEW short ASCII results directory')
    p.add_argument('--vivado', help='Vivado executable (or use VIVADO_BAT/PATH)')
    p.add_argument('--capture-dir', help='Successful capture directory, required for report mode')
    p.add_argument('--saif-manifest', type=Path,
                   help='Verified repaired-SAIF JSON manifest for report mode')
    p.add_argument('--probe-dir', help='Successful v2 probe directory, required for capture/all')
    p.add_argument('--min-free-gib', type=float, default=None)
    a = p.parse_args()
    build = Path(a.build).resolve()
    minimum = a.min_free_gib if a.min_free_gib is not None else (5 if a.mode=='report' else 40)
    if minimum < 0:
        raise RuntimeError('min-free-gib must be nonnegative')
    info = preflight(build, a.mode, minimum)
    if a.mode == 'check':
        print('CHECK OK. No Vivado run launched. Capture recommends >=40 GiB free.')
        return
    vivado_name = a.vivado or os.environ.get('VIVADO_BAT') or shutil.which('vivado')
    if not vivado_name:
        raise RuntimeError('Vivado not found; use --vivado, VIVADO_BAT or PATH')
    vivado = Path(vivado_name).resolve()
    if not vivado.is_file():
        raise RuntimeError('Vivado not found; use --vivado or VIVADO_BAT')
    out = Path(a.out or (build.parent / ('n3_power_' + datetime.now().strftime('%Y%m%d_%H%M%S')))).resolve()
    if out.exists() or not str(out).isascii() or len(str(out)) > 48:
        raise RuntimeError('Output must be NEW, ASCII and <=48 characters: ' + str(out))
    capture = out
    probe = out
    if a.mode in ('all', 'capture'):
        if not a.probe_dir:
            raise RuntimeError('Run probe first, then supply --probe-dir DIR containing PROBE_OK.tcl')
        probe = Path(a.probe_dir).resolve()
        if not (probe / 'PROBE_OK.tcl').is_file():
            raise RuntimeError('No PROBE_OK.tcl in --probe-dir; an unfinished probe is not accepted')
    if a.mode == 'report':
        if not a.capture_dir:
            raise RuntimeError('report requires --capture-dir from a successful capture')
        capture = Path(a.capture_dir).resolve()
        if not (capture / 'CAPTURE_OK.tcl').is_file():
            raise RuntimeError('No CAPTURE_OK.tcl; incomplete/old SAIF is not accepted')
    report_saif = report_sha = None
    if a.saif_manifest:
        if a.mode != 'report':
            raise RuntimeError('--saif-manifest is only valid for report mode')
        manifest = json.loads(a.saif_manifest.read_text(encoding='utf-8'))
        receipt = (capture / 'CAPTURE_OK.tcl').read_text(encoding='utf-8')
        match = re.search(r'^set capture_saif (.+)$', receipt, re.M)
        original = Path(match[1].strip('{}')).resolve() if match else None
        match = re.search(r'^set capture_saif_hash ([0-9a-f]{64})$', receipt, re.M)
        original_sha = match[1] if match else None
        if (manifest.get('method') != 'xsim-null-net-name-v1'
                or Path(manifest['source']).resolve() != original
                or manifest.get('source_sha256') != original_sha
                or manifest.get('renamed_names', 0) < 1):
            raise RuntimeError('SAIF repair manifest does not match this capture')
        report_saif = Path(manifest['output']).resolve(strict=True)
        report_sha = manifest['output_sha256']
        if report_saif.stat().st_size != manifest['output_bytes'] or digest(report_saif) != report_sha:
            raise RuntimeError('Repaired SAIF size/hash differs from its manifest')
        print(f'REPAIRED SAIF VERIFIED: {report_saif} names={manifest["renamed_names"]}', flush=True)
    if shutil.disk_usage(out.parent).free / 2**30 < minimum:
        raise RuntimeError('Insufficient free space on output drive')
    out.mkdir()
    info.update({'mode': a.mode, 'output': str(out), 'capture': str(capture), 'probe': str(probe),
                 'saif_registration': 'v2: no type/signal filters; depth-3 subtrees; batches of 1024',
                 'protocol': '100MHz; first 16 different tiles; 9216-cycle warmup; 32768-cycle SAIF window'})
    (out / 'preflight.json').write_text(json.dumps(info, indent=2) + '\n', encoding='utf-8')
    print(f'NEW RESULTS: {out}', flush=True)
    try:
        if a.mode == 'probe':
            invoke(vivado, 'probe', build, out, out)
        if a.mode in ('all', 'capture'):
            invoke(vivado, 'capture', build, out, probe)
        if a.mode in ('all', 'report'):
            invoke(vivado, 'report', build, out, capture, report_saif, report_sha)
    except Exception:
        if (capture / 'CAPTURE_OK.tcl').exists():
            retry = runner_command('report', build, '--capture-dir', capture, vivado=vivado)
            print(f'Capture retained. Retry reports with: {retry}', flush=True)
        raise
    if a.mode == 'probe':
        next_command = runner_command('all', build, '--probe-dir', out, vivado=vivado)
        print(f'PROBE COMPLETE: {out}\nNext: {next_command}')
    else:
        print(f'COMPLETE: {out}\nPower is a Vivado estimate, not measured board power. Activity coverage needs review.')


if __name__ == '__main__':
    # Vivado/Windows may mix console encodings; logging must not abort simulation.
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(errors='replace')
        sys.stderr.reconfigure(errors='replace')
    try:
        main()
    except Exception as exc:
        print('ERROR: ' + str(exc), file=sys.stderr)
        sys.exit(1)
