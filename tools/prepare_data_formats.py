#!/usr/bin/env python3
"""Validate four dataset MAT pairs; optionally convert upstream HongHu NPY files.

Existing MAT files are never overwritten. No normalization, PCA, dtype cast,
resizing, label remapping, or axis permutation is performed.
"""
import argparse
import hashlib
import json
from pathlib import Path


DATASETS = {
    "UP": ("PaviaU", "PaviaU_gt", "paviaU", "paviaU_gt", 9),
    "HanChuan": ("WHU_Hi_HanChuan", "WHU_Hi_HanChuan_gt", "WHU_Hi_HanChuan", "WHU_Hi_HanChuan_gt", 16),
    "HongHu": ("WHU_Hi_HongHu", "WHU_Hi_HongHu_gt", "WHU_Hi_HongHu", "WHU_Hi_HongHu_gt", 22),
    "Houston": ("Houston", "Houston_GT", "Houston", "Houston_GT", 15),
}


def validate_arrays(data, gt, name, np):
    classes = DATASETS[name][4]
    if data.ndim != 3 or gt.ndim != 2 or data.shape[:2] != gt.shape:
        raise ValueError("Expected HxWxB data and HxW labels; got {} and {}".format(data.shape, gt.shape))
    if not np.issubdtype(data.dtype, np.number) or not np.issubdtype(gt.dtype, np.number):
        raise ValueError("Dataset arrays must have numeric dtypes")
    # Limit temporary allocation on full-resolution scenes.
    if any(not np.isfinite(data[row:row+32]).all() for row in range(0, data.shape[0], 32)):
        raise ValueError("Image contains NaN/Inf")
    if not np.isfinite(gt).all() or not np.equal(gt, np.floor(gt)).all():
        raise ValueError("Ground truth must contain finite integer labels")
    labels = np.unique(gt)
    if np.any(labels < 0) or not np.array_equal(labels[labels > 0], np.arange(1, classes+1)):
        raise ValueError("Expected background 0 and foreground 1..{}; got {}".format(classes, labels.tolist()))
    return {"dataset": name, "data_shape": list(data.shape), "label_shape": list(gt.shape),
            "data_dtype": str(data.dtype), "label_dtype": str(gt.dtype), "classes": classes}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert_honghu(root, np, sio):
    folder = root / "HongHu"
    image, label, image_key, label_key, _ = DATASETS["HongHu"]
    src = [folder / (base+".npy") for base in (image, label)]
    dst = [folder / (base+".mat") for base in (image, label)]
    manifest = folder / "npy_to_mat_manifest.json"
    for path in dst + [manifest]:
        if path.exists():
            raise FileExistsError("Refusing to overwrite: " + str(path))
    for path in src:
        if not path.is_file():
            raise FileNotFoundError(str(path))
    arrays = [np.load(str(path), mmap_mode="r", allow_pickle=False) for path in src]
    details = validate_arrays(arrays[0], arrays[1], "HongHu", np)
    if any(array.nbytes >= 2_000_000_000 for array in arrays):
        raise ValueError("Array exceeds conservative MAT v5 size limit; supply dataset MAT files directly")
    created = []
    try:
        for path, key, array in zip(dst, (image_key, label_key), arrays):
            with path.open("xb") as handle:
                created.append(path)
                sio.savemat(handle, {key: array}, do_compression=True)
        details["files"] = [{"source": x.name, "source_sha256": sha256(x),
                              "output": y.name, "output_sha256": sha256(y)} for x,y in zip(src,dst)]
        with manifest.open("x", encoding="utf-8") as handle:
            created.append(manifest)
            json.dump(details, handle, indent=2)
            handle.write("\n")
    except Exception:
        for path in created:
            path.unlink()
        raise
    return details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--datasets", default="UP,HanChuan,HongHu,Houston")
    parser.add_argument("--convert-honghu", action="store_true", help="create missing HongHu MAT pair from upstream NPY pair")
    args = parser.parse_args()
    names = [x.strip() for x in args.datasets.split(",") if x.strip()]
    if not names or any(x not in DATASETS for x in names):
        parser.error("--datasets must be comma-separated names from " + ",".join(DATASETS))
    import numpy as np
    from scipy import io as sio
    root = args.data_root.expanduser().resolve()
    if args.convert_honghu:
        print(json.dumps({"conversion": convert_honghu(root, np, sio)}, indent=2))
    for name in names:
        image, label, image_key, label_key, _ = DATASETS[name]
        data_path = root/name/(image+".mat")
        gt_path = root/name/(label+".mat")
        data = sio.loadmat(str(data_path), variable_names=[image_key])[image_key]
        gt = sio.loadmat(str(gt_path), variable_names=[label_key])[label_key]
        print(json.dumps(validate_arrays(data, gt, name, np), indent=2))
        del data, gt


if __name__ == "__main__":
    main()
