#!/usr/bin/env python3
"""Collect a bounded, read-only reproducibility snapshot (Python 3.8+).

Run with the original mambahsi Python interpreter. The project, environment and
GPU settings are never changed. Only --output-dir receives new files.
"""
import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from urllib.parse import urlsplit, urlunsplit
import zipfile


SOURCES = (
    "train_mambahsi_spatial_split_128_dense.py", "mambahsi_ablation_model.py",
    "train_mambahsi_spatial_split_dense_qat.py", "qat_forensics.py",
    "both_FPGA_single_qat_source.py", "ssm_error_ablation.py", "ssm_d_path.py",
    "train_shared_a_frozen_pair.py", "analyze_alog_dynamics.py",
    "run_current_a_gpu_experiments.py", "run_qat_stability_sweep.py",
    "replay_qat_jump.py", "run_max_init_qat_fpga.py",
    "run_qat_eval_batch1_four_datasets.py", "run_fpga_qat_eval1_four_datasets.py",
    "run_fpga_error_sweep.py", "run_ssm_error_ablation.py", "diagnose_qat_batch.py",
    "benchmark_gpu_batch1_dense16.py", "fixed_gpu_backend.py",
    "run_fixed_gpu_four_datasets.py", "benchmark_up_gpu_power.py",
    "gpu_power_monitor.py", "run_gpu_current_four_datasets.sh",
    "run_shared_a_current_four_datasets.sh", "run_fixed_gpu_four_datasets.sh",
    "run_up_gpu_power.sh", "run_qat_eval1_four_datasets.sh",
    "run_fpga_qat_eval1_four_datasets.sh",
    "utils/data_load_operate.py", "utils/evaluation.py", "utils/visual_predict.py",
    "utils/setup_logger.py", "utils/HSICommonUtils.py", "utils/Loss.py",
) + tuple("ablation_scripts/" + name + ".sh" for name in (
    "_common", "00_baseline_current", "01_reference_original_matched",
    "02_reference_original_full", "05_fusion_mean", "06_fusion_softmax",
    "07_skip_scale_1", "08_skip_scale_0", "09_restore_z", "10_restore_D",
    "11_A_per_channel", "12_norm_gn_path", "13_activation_silu", "14_head_dim_32",
    "15_head_dim_128", "16_token_num_2", "17_token_num_8", "18_d_state_8",
    "19_d_state_32", "20_norm_gn_activation_silu", "21_restore_z_D",
))
CHECKPOINTS = {"best_model.pth", "best_model.pt", "best_qat.pth",
               "best_qat_foldaware.pth", "best_qat_deploy.pth"}
TEXT_SUFFIXES = {".json", ".csv", ".log"}
SKIP_DIRS = {".git", ".ssh", ".aws", ".cache", "__pycache__", "wandb",
             "node_modules", "site-packages", "venv", ".venv"}
MIB = 1024 * 1024


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(MIB), b""):
            h.update(block)
    return h.hexdigest()


def redact(text):
    # Keep filenames and numeric experiment records intact; never dump env vars.
    def clean_url(match):
        try:
            u = urlsplit(match.group())
            host = u.hostname or ""
            if u.port:
                host += ":" + str(u.port)
            return urlunsplit((u.scheme, host, u.path, "", ""))
        except ValueError:
            return "[URL REDACTED]"
    text = re.sub(r"https?://[^\s\"'<>]+", clean_url, text)
    text = re.sub(r'''(?i)((?:["']?)(?:api[_-]?key|access[_-]?token|password|secret)(?:["']?)\s*[=:]\s*)(?:"[^"]*"|'[^']*'|[^\s,;}\]]+)''',
                  r'\1"[REDACTED]"', text)
    return text


def execute(command, timeout=45):
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    env["PIP_NO_CACHE_DIR"] = "1"
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, timeout=timeout, env=env)
        return {"command": command, "returncode": result.returncode,
                "stdout": redact(result.stdout), "stderr": redact(result.stderr)}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"command": command, "error": redact(str(error))}


RUNTIME_PROBE = r'''
import importlib.util, json, platform, sys
out = {"python":sys.version, "executable":sys.executable, "platform":platform.platform()}
try:
    import torch
    out.update(torch=torch.__version__, torch_cuda=torch.version.cuda,
               cudnn=torch.backends.cudnn.version(), cuda_available=torch.cuda.is_available())
    out["numeric_flags"] = {
        "cudnn_benchmark":torch.backends.cudnn.benchmark,
        "cudnn_deterministic":torch.backends.cudnn.deterministic,
        "cudnn_allow_tf32":getattr(torch.backends.cudnn,"allow_tf32",None),
        "matmul_allow_tf32":getattr(torch.backends.cuda.matmul,"allow_tf32",None),
        "deterministic_algorithms":torch.are_deterministic_algorithms_enabled()}
    out["devices"] = [{"index":i,"name":torch.cuda.get_device_name(i),
        "capability":list(torch.cuda.get_device_capability(i)),
        "total_memory":torch.cuda.get_device_properties(i).total_memory}
        for i in range(torch.cuda.device_count())]
except Exception as e:
    out["torch_error"] = repr(e)
for name in ["mamba_ssm", "selective_scan_cuda", "causal_conv1d", "causal_conv1d_cuda"]:
    try:
        spec=importlib.util.find_spec(name)
        out[name] = {"origin":spec.origin,"locations":list(spec.submodule_search_locations or [])} if spec else None
    except Exception as e:
        out[name] = {"error":repr(e)}
print(json.dumps(out, indent=2))
'''


def capture_environment(destination):
    records = {}
    records["runtime"] = execute([sys.executable, "-B", "-c", RUNTIME_PROBE], timeout=60)
    records["pip_list"] = execute([sys.executable, "-B", "-m", "pip", "list", "--format=json"])
    for key, command in (
        ("conda_list", ["conda", "list", "--json"]),
        ("nvidia_smi", ["nvidia-smi"]),
        ("gpu_details", ["nvidia-smi", "--query-gpu=index,name,uuid,driver_version,pci.bus_id,power.limit", "--format=csv"]),
        ("nvcc", ["nvcc", "--version"]),
    ):
        records[key] = execute(command)
    # Hash the installed scan interface and extension, without copying binaries.
    provenance = []
    try:
        runtime = json.loads(records["runtime"].get("stdout", "{}"))
    except ValueError:
        runtime = {}
    for name in ("mamba_ssm", "selective_scan_cuda", "causal_conv1d", "causal_conv1d_cuda"):
        info = runtime.get(name) or {}
        candidates = []
        if info.get("origin"):
            candidates.append(Path(info["origin"]))
        for location in info.get("locations", []):
            candidates.extend([Path(location)/"ops"/"selective_scan_interface.py",
                               Path(location)/"modules"/"mamba_simple.py"])
        for path in candidates:
            if path.is_file():
                provenance.append({"module":name, "path":str(path),
                                   "size":path.stat().st_size,"sha256":digest(path)})
    distributions = []
    for name in ("torch", "mamba-ssm", "causal-conv1d", "triton", "nvidia-ml-py"):
        try:
            package = importlib.metadata.distribution(name)
            item = {"name":name,"version":package.version}
            direct = package.read_text("direct_url.json")
            if direct:
                item["direct_url"] = json.loads(redact(direct))
            distributions.append(item)
        except importlib.metadata.PackageNotFoundError:
            distributions.append({"name":name,"installed":False})
    records["scan_provenance"] = provenance
    records["key_distributions"] = distributions
    destination.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def allowed_artifact(path, args):
    lower = path.name.lower()
    if any(token in lower for token in ("credential", "password", "secret", "token.json")):
        return False
    if path.suffix.lower() in TEXT_SUFFIXES:
        return True
    if args.include_checkpoints and path.name in CHECKPOINTS:
        return True
    if args.include_repro_artifacts and path.suffix.lower() == ".npz":
        return any(token in lower for token in ("preprocess", "split", "indices", "index"))
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path,
                        help="Original MambaHSI project directory (read only)")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="Where the new snapshot directory and ZIP are written")
    parser.add_argument("--artifact-root", action="append", type=Path, default=[],
                        help="Explicit result directory to collect; repeat for multiple directories")
    parser.add_argument("--source-file", action="append", type=Path, default=[],
                        help="Additional explicit .py/.sh source, relative to project root")
    parser.add_argument("--include-checkpoints", action="store_true")
    parser.add_argument("--include-repro-artifacts", action="store_true")
    parser.add_argument("--max-file-mib", type=float, default=16)
    parser.add_argument("--max-total-mib", type=float, default=256)
    parser.add_argument("--skip-environment", action="store_true",
                        help="For a collection smoke test; real server collection should omit this")
    args = parser.parse_args()
    project = args.project_root.expanduser().resolve()
    if not project.is_dir():
        parser.error("--project-root must be an existing directory")
    if args.max_file_mib <= 0 or args.max_total_mib <= 0:
        parser.error("Size limits must be positive")
    sources = [Path(p) for p in SOURCES]
    for path in args.source_file:
        if path.is_absolute() or ".." in path.parts or path.suffix.lower() not in (".py", ".sh"):
            parser.error("--source-file requires a relative .py/.sh path inside the project")
        sources.append(path)
    artifact_roots = []
    for path in args.artifact_root:
        path = path.expanduser()
        path = (project/path if not path.is_absolute() else path).resolve()
        if not path.is_dir():
            parser.error("Artifact directory does not exist: " + str(path))
        artifact_roots.append(path)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = Path(tempfile.mkdtemp(prefix="FPGA-MambaHSI-server-"+stamp+"-", dir=str(output)))
    manifest = {"schema":1, "created_utc":stamp, "project_root":str(project),
                "artifact_roots":[str(p) for p in artifact_roots],
                "files":[], "skipped":[], "limits_mib":{"file":args.max_file_mib,"total":args.max_total_mib},
                "include_checkpoints":args.include_checkpoints,
                "include_repro_artifacts":args.include_repro_artifacts}
    copied_bytes = 0

    def copy_file(source, relative, sanitize=False):
        nonlocal copied_bytes
        if source.is_symlink():
            manifest["skipped"].append({"path":str(source),"reason":"symlink"})
            return
        if not source.is_file():
            manifest["skipped"].append({"path":str(source),"reason":"missing"})
            return
        size = source.stat().st_size
        if size > args.max_file_mib*MIB or copied_bytes+size > args.max_total_mib*MIB:
            manifest["skipped"].append({"path":str(source),"reason":"size_limit","bytes":size})
            return
        dest = target/relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        if sanitize:
            original = source.read_bytes()
            try:
                cleaned = redact(original.decode("utf-8"))
            except UnicodeDecodeError:
                manifest["skipped"].append({"path":str(source),"reason":"not_utf8_text"})
                return
            dest.write_text(cleaned, encoding="utf-8")
        else:
            shutil.copyfile(str(source), str(dest))
        stored_size = dest.stat().st_size
        copied_bytes += stored_size
        source_hash, stored_hash = digest(source), digest(dest)
        manifest["files"].append({"source":str(source), "path":str(relative),
                                  "size":stored_size, "sha256":stored_hash,
                                  "source_sha256":source_hash,"redacted":source_hash != stored_hash})

    for path in sorted(set(sources)):
        full = project/path
        # A symlink in any parent must not escape the named source tree.
        if project not in full.resolve().parents:
            manifest["skipped"].append({"path":str(full),"reason":"outside_project"})
            continue
        copy_file(full, Path("source")/path)
    for index, root in enumerate(artifact_roots):
        for parent, folders, files in os.walk(str(root), followlinks=False):
            parent = Path(parent)
            folders[:] = sorted(d for d in folders if d not in SKIP_DIRS
                                and not (parent/d).is_symlink()
                                and (parent/d).resolve() != output
                                and (parent/d).resolve() != target)
            if parent == output or output in parent.parents:
                folders[:] = []
                continue
            for name in sorted(files):
                path = parent/name
                if allowed_artifact(path,args):
                    relative = Path("artifacts")/("%02d_%s"%(index,root.name))/path.relative_to(root)
                    copy_file(path, relative, sanitize=path.suffix.lower() in TEXT_SUFFIXES)
    if not args.skip_environment:
        capture_environment(target/"environment.json")
    manifest["collected_bytes"] = copied_bytes
    (target/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    # Hash generated metadata as well as copied data. SHA256SUMS excludes itself.
    all_files = sorted(p for p in target.rglob("*") if p.is_file())
    (target/"SHA256SUMS").write_text("".join(digest(p)+"  "+p.relative_to(target).as_posix()+"\n"
                                             for p in all_files),encoding="utf-8")
    archive = target.with_suffix(".zip")
    with zipfile.ZipFile(str(archive),"w",compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(p for p in target.rglob("*") if p.is_file()):
            z.write(str(path),str(Path(target.name)/path.relative_to(target)))
    archive.with_suffix(".zip.sha256").write_text(digest(archive)+"  "+archive.name+"\n",encoding="utf-8")
    print("Collected %d files (%.2f MiB); %d missing/skipped entries." %
          (len(manifest["files"]),copied_bytes/MIB,len(manifest["skipped"])))
    print("Snapshot: "+str(target))
    print("Archive:  "+str(archive))
    print("Check manifest.json for missing files/size limits before copying the ZIP.")


if __name__ == "__main__":
    main()
