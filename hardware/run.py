"""Portable hardware runner. No operation programs a board or overwrites a build."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'tools'))
from verify import verify, digest


def tcl(value):
    value = str(value).replace('\\', '/')
    if any(c in value for c in '{}\n\r'):
        raise ValueError('Unsupported Tcl path characters')
    return '{' + value + '}'


def command(exe, script, *args):
    return [exe, '-mode', 'batch', '-nojournal', '-nolog', '-source', str(script),
            '-tclargs', *map(str, args)]


def find_vivado(requested):
    candidate = requested or os.environ.get('VIVADO_BAT') or shutil.which('vivado')
    if not candidate or not Path(candidate).is_file():
        raise RuntimeError('Vivado not found. Use --vivado PATH or VIVADO_BAT.')
    return str(Path(candidate).resolve())


def invoke(exe, script, args, cwd, log_name):
    path = cwd / log_name
    # Timestamp makes repeated diagnostics non-destructive.
    import time
    if path.exists():
        path = cwd / (path.stem + '_' + str(time.time_ns()) + path.suffix)
    cmd = command(exe, script, *args)
    print(subprocess.list2cmdline(cmd), flush=True)
    with path.open('x', encoding='utf-8') as log:
        proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for raw in iter(proc.stdout.readline, b''):
            line = raw.decode('utf-8', 'replace')
            log.write(line); log.flush()
            print(line, end='', flush=True)
        rc = proc.wait()
    if rc:
        raise RuntimeError(f'Vivado exit {rc}; retained log: {path}')


def relocate(obj, mappings):
    if isinstance(obj, dict):
        return {k: relocate(v, mappings) for k, v in obj.items()}
    if isinstance(obj, list):
        return [relocate(v, mappings) for v in obj]
    return mappings.get(obj, obj) if isinstance(obj, str) else obj


def stage(profile, output, vendor=None, enforce_short_path=True):
    source_hash = verify(ROOT)
    output = Path(output).resolve()
    if output.exists():
        raise RuntimeError('Choose a NEW directory; existing builds are preserved: ' + str(output))
    if enforce_short_path and (not str(output).isascii() or len(str(output)) > 48):
        raise RuntimeError('Use an ASCII build path of at most 48 characters, e.g. D:/fpga/n3_v1')
    vendor_files = []
    if profile == 'board':
        vendor = Path(vendor) if vendor is not None else ROOT / 'vendor/hdmi'
        for item in json.loads((ROOT/'config/vendor_requirements.json').read_text(encoding='utf8')):
            path = Path(vendor) / item['name']
            if not path.is_file() or digest(path) != item['sha256']:
                raise RuntimeError('Vendor source missing or differs from tested version: ' + str(path))
            vendor_files.append(path)
    output.mkdir(parents=True)
    package = output / 'package'
    shutil.copytree(ROOT, package, ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    assert verify(package) == source_hash
    config = json.loads((package/'config/profiles.json').read_text(encoding='utf8'))[profile]
    paths = [package / p for p in config['rtl']]
    for path in vendor_files:
        target = output / 'vendor' / path.name
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(path, target); paths.append(target)
    catalog = json.loads((package/'config/ip_catalog.json').read_text(encoding='utf8'))
    ip_paths=[]
    for item in catalog:
        if profile != 'board' and item['board_only']:
            continue
        obj=json.loads((package/item['path']).read_text(encoding='utf8'))
        mappings={old:(package/new).as_posix() for old,new in item['relocations'].items()}
        obj=relocate(obj,mappings)
        dest=output/'ip'/item['name']/(item['name']+'.xci')
        dest.parent.mkdir(parents=True)
        # Vivado regenerates products locally, never back into the source repository.
        generated = '../../generated_ip/' + item['name']
        obj['ip_inst']['gen_directory'] = generated
        runtime = obj['ip_inst']['parameters']['runtime_parameters']
        for entry in runtime['OUTPUTDIR']:
            entry['value'] = generated
        dest.write_text(json.dumps(obj,indent=2)+'\n',encoding='utf8');ip_paths.append(dest)
    setup=[f'set hw_profile {tcl(profile)}',f'set nf_out {tcl(output)}',
           f'set nf_python {tcl(sys.executable)}',f'set nf_package {tcl(package)}',
           f'set nf_method {tcl(profile)}','set nf_period_ns 10',
           f'set nf_source_hash {tcl(source_hash)}',
           'set hw_rtl [list '+ ' '.join(map(tcl,paths))+']',
           'set hw_ips [list '+ ' '.join(map(tcl,ip_paths))+']']
    if profile=='board':
        bd_obj=json.loads((package/config['bd']).read_text(encoding='utf8'))
        bd=output/'bd/design_1/design_1.bd';bd.parent.mkdir(parents=True)
        bd_obj['design']['design_info']['gen_directory']='../../generated_bd/design_1'
        bd.write_text(json.dumps(bd_obj,indent=2)+'\n',encoding='utf8')
        setup += [f'set hw_bd {tcl(bd)}','set hw_xdc [list '+' '.join(tcl(package/p) for p in config['constraints'])+']']
    (output/'config.tcl').write_text('\n'.join(setup)+'\n',encoding='utf8')
    receipt={'profile':profile,'source_sha256':source_hash,'period_ns':10,
             'project':'pcie_network_hdmi.xpr' if profile=='board' else 'nf_fullnet.xpr',
             'vendor_files':[{ 'name':p.name,'sha256':digest(p)} for p in vendor_files],
             'status':'STAGED_NOT_VALIDATED_BY_VIVADO'}
    (output/'build_config.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8')
    print('STAGED:', output, '\nSOURCE:',source_hash)
    return output


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest='action',required=True)
    sub.add_parser('check')
    p=sub.add_parser('prepare');p.add_argument('profile',choices=['board','N0','N1','N2','N3']);p.add_argument('build',type=Path)
    p.add_argument('--vendor-dir',type=Path,help='HDMI RTL directory (default: hardware/vendor/hdmi)');p.add_argument('--stage-only',action='store_true');p.add_argument('--vivado')
    for name in ['create','sim','build']:
        p=sub.add_parser(name);p.add_argument('build',type=Path);p.add_argument('--vivado');p.add_argument('--dry-run',action='store_true')
    a=ap.parse_args()
    if a.action=='check':
        print('Hardware SHA256:',verify(ROOT));return
    if a.action=='prepare':
        exe=None if a.stage_only else find_vivado(a.vivado)
        out=stage(a.profile,a.build,a.vendor_dir)
        if exe:invoke(exe,out/'package/scripts/create.tcl',[out],out,'prepare_console.log')
        return
    out=a.build.resolve();cfg=json.loads((out/'build_config.json').read_text(encoding='utf8'))
    if verify(out/'package')!=cfg['source_sha256']:raise RuntimeError('Frozen build source mismatch')
    script=out/'package/scripts'/(a.action+'.tcl')
    if a.dry_run:
        print(subprocess.list2cmdline(command(a.vivado or 'vivado',script,out)))
        return
    invoke(find_vivado(a.vivado),script,[out],out,a.action+'_console.log')


if __name__=='__main__':
    try:main()
    except Exception as exc:
        print('ERROR:',exc,file=sys.stderr);sys.exit(1)
