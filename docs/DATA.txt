# Datasets and saved artifacts

Use the original full-resolution scenes. Put the image and label files beneath
a common data root, then pass `--data-root /path/to/data` to the numbered entry
scripts. No raw dataset, checkpoint or preprocessed image is included in this
software package.

```text
data/
├── UP/
│   ├── PaviaU.mat
│   └── PaviaU_gt.mat
├── HanChuan/
│   ├── WHU_Hi_HanChuan.mat
│   └── WHU_Hi_HanChuan_gt.mat
├── HongHu/
│   ├── WHU_Hi_HongHu.mat
│   └── WHU_Hi_HongHu_gt.mat
└── Houston/
    ├── Houston.mat
    └── Houston_GT.mat
```

| Dataset | Image variable | Label variable | Classes |
|---|---|---|---:|
| UP | `paviaU` | `paviaU_gt` | 9 |
| HanChuan | `WHU_Hi_HanChuan` | `WHU_Hi_HanChuan_gt` | 16 |
| HongHu | `WHU_Hi_HongHu` | `WHU_Hi_HongHu_gt` | 22 |
| Houston | `Houston` | `Houston_GT` | 15 |

Images have shape H × W × bands; labels have shape H × W. Label 0 is background,
and labels 1 through C identify the classes. The FP32 loader uses these variable
names exactly. The QAT loader accepts some additional key/axis variants, but the
layout above works for both loaders and benchmarks. A pre-cropped or PCA-reduced
`PaviaU_128.mat` or `PaviaU_16.mat` is not an interchangeable raw input.

Obtain the scenes through the dataset links in the
[MambaHSI repository](https://github.com/li-yapeng/MambaHSI#data-preparation),
following each dataset's access and citation terms. Its example uses **NPY** for
HongHu; the supplied FP32 and QAT sources both use **MAT**. If the downloaded
HongHu files are NPY, convert that pair once:

```bash
python tools/prepare_data_formats.py --data-root ./data --datasets HongHu --convert-honghu
```

This reads `HongHu/WHU_Hi_HongHu.npy` and `WHU_Hi_HongHu_gt.npy`, creates the MAT
files/keys listed above and records input/output SHA-256 hashes in
`HongHu/npy_to_mat_manifest.json`. The converter preserves values, dtypes and
axis order and refuses to overwrite any existing outputs. If MAT files are
already present, use validation without conversion:

```bash
python tools/prepare_data_formats.py --data-root ./data
```

This checks MAT keys, dimensions, numeric finiteness and class labels. It reads
the full arrays and therefore needs RAM for one complete scene at a time. Use
`--datasets UP` to validate a single pair. Standard MAT v5 files work with both
loaders; only the QAT loader has an h5py fallback for MATLAB v7.3, so a v7.3-only
input is not the common FP32/QAT interchange format.

## FP32 → QAT → simulation

The FP32 experiment directory contains the checkpoint, the saved train-only
preprocessing transform, split information, configuration and seed-specific
evaluation artifacts. QAT and the FPGA simulator must reuse these artifacts from
the **same configuration and seed**. Do not refit PCA or silently regenerate
train/validation/test masks when replaying saved results.

Pass the complete configuration directory to `--fp32-template`, with
`{dataset}` as its optional placeholder, and the seed run directory to
`--qat-template`, with `{dataset}` and `{seed}` placeholders. The numbered
scripts have portable defaults under the repository's `results/`; existing
server directories can be used directly via these overrides. The project-specific
result folder naming and detailed commands are documented in the main README.

The published artifacts should include `result.json`, configuration files,
split/seed indices, `train_only_preprocess.npz` and the matching selected
checkpoint for each reported run. Large artifacts can be distributed separately
from Git source. A complete artifact manifest should state the dataset version,
file hashes and the associated environment report.
