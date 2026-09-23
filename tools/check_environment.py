#!/usr/bin/env python3
"""Inspect imports and optionally execute a real CUDA selective-scan operation.

Python >= 3.8. This does not install packages or modify the environment.
"""
import argparse
import importlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path


REQUIRED = {
    "numpy": "numpy", "scipy": "scipy", "sklearn": "scikit-learn",
    "matplotlib": "matplotlib", "torch": "torch",
}
OPTIONAL = {
    "h5py": "h5py", "spectral": "spectral", "pynvml": "nvidia-ml-py",
    "mamba_ssm": "mamba-ssm", "calflops": "calflops",
}


def inspect_import(module, distribution):
    try:
        imported = importlib.import_module(module)
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            version = getattr(imported, "__version__", "unknown")
        return {"ok": True, "version": version,
                "module_file": getattr(imported, "__file__", None)}
    except Exception as exc:
        return {"ok": False, "error": type(exc).__name__ + ": " + str(exc)}


def check_scan(torch, device):
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
    # Short FP32 forward/backward test, independent of datasets/checkpoints.
    torch.manual_seed(0)
    u = (torch.randn(2, 4, 8, device=device) * 0.1).requires_grad_()
    dt = (torch.randn(2, 4, 8, device=device) * 0.1).requires_grad_()
    poles = (-torch.rand(4, 3, device=device) - 0.1).requires_grad_()
    b = (torch.randn(2, 3, 8, device=device) * 0.1).requires_grad_()
    c = (torch.randn(2, 3, 8, device=device) * 0.1).requires_grad_()
    d = torch.ones(4, device=device, requires_grad=True)
    inputs = (u, dt, poles, b, c, d)
    result = selective_scan_fn(*inputs, delta_softplus=True)
    reference = selective_scan_ref(*inputs, delta_softplus=True)
    torch.testing.assert_close(result, reference, rtol=1e-3, atol=1e-5)
    grads = torch.autograd.grad(result.sum(), inputs)
    ref_grads = torch.autograd.grad(reference.sum(), inputs)
    for grad, ref_grad in zip(grads, ref_grads):
        torch.testing.assert_close(grad, ref_grad, rtol=1e-3, atol=1e-5)
    torch.cuda.synchronize(device)
    return {"ok": True, "forward_max_abs_error": float((result-reference).abs().max()),
            "backward_max_abs_error": max(float((x-y).abs().max()) for x,y in zip(grads,ref_grads)),
            "backend": "mamba_ssm selective_scan_cuda"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", help="cpu or CUDA logical device, e.g. cuda:1")
    parser.add_argument("--check-scan", action="store_true", help="require and run optimized CUDA scan forward/backward")
    parser.add_argument("--json", type=Path, help="optional report path, must not already exist")
    args = parser.parse_args()
    if args.json and args.json.exists():
        parser.error("Report exists: " + str(args.json))
    report = {"python": sys.version, "platform": platform.platform(), "device": args.device,
              "required_imports": {}, "optional_imports": {}, "errors": []}
    for target, mapping in (("required_imports", REQUIRED), ("optional_imports", OPTIONAL)):
        for module, distribution in mapping.items():
            report[target][module] = inspect_import(module, distribution)
            if target == "required_imports" and not report[target][module]["ok"]:
                report["errors"].append("Required import failed: " + module)
    if report["required_imports"]["torch"]["ok"]:
        import torch
        report["torch"] = {"version": torch.__version__, "cuda_build": torch.version.cuda,
                           "cuda_available": torch.cuda.is_available(),
                           "cudnn": torch.backends.cudnn.version(),
                           "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
                           "tf32_cudnn": torch.backends.cudnn.allow_tf32}
        try:
            device = torch.device(args.device)
            if device.type not in ("cpu", "cuda"):
                raise ValueError("Use cpu or cuda:N")
            if device.type == "cuda":
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA unavailable; check nvidia-smi, driver/library compatibility and CUDA PyTorch build")
                torch.ones(1, device=device).sum().item()
                report["torch"]["gpu"] = torch.cuda.get_device_name(device)
                report["torch"]["compute_capability"] = list(torch.cuda.get_device_capability(device))
            if args.check_scan:
                if device.type != "cuda":
                    raise ValueError("--check-scan requires --device cuda:N; a CPU fallback does not validate GPU timing")
                report["selective_scan"] = check_scan(torch, device)
        except Exception as exc:
            report["errors"].append(type(exc).__name__ + ": " + str(exc))
    elif args.check_scan:
        report["errors"].append("Cannot test scan without PyTorch")
    report["ok"] = not report["errors"]
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with args.json.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
