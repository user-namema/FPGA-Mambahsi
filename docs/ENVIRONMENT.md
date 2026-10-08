# Environment setup

Prerequisites: Linux x86_64, NVIDIA GPU/driver, Conda, CUDA 11.7 Toolkit with `nvcc`, and a C++ compiler.

## 1. Configure your paths

```bash
export PROJECT_ROOT="/absolute/path/to/FPGA-MambaHSI"
export RESULTS_ROOT="$PROJECT_ROOT/results"
export CUDA_HOME="/absolute/path/to/cuda-11.7"
export ENV_NAME="fpga-mambahsi"
export DEVICE="cuda:0"
export PATH="$CUDA_HOME/bin:$PATH"
cd "$PROJECT_ROOT"
mkdir -p "$RESULTS_ROOT"
nvidia-smi
nvcc --version
c++ --version
```

`CUDA_HOME`: Toolkit installation directory. `DEVICE`: logical index of the GPU visible to this process.

## 2. Install Python and PyTorch

```bash
conda create -n "$ENV_NAME" python=3.9 pip -y
conda activate "$ENV_NAME"
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 \
  pytorch-cuda=11.7 'numpy<2' -c pytorch -c nvidia -y
```

Installation references: [PyTorch](https://pytorch.org/get-started/previous-versions/), [Mamba 1.2.0](https://github.com/state-spaces/mamba/blob/v1.2.0/README.md).

## 3. Install project dependencies and selective scan

```bash
python -m pip install -c environment/constraints-torch113.txt -r environment/requirements.txt
python -m pip install -c environment/constraints-torch113.txt --no-build-isolation mamba-ssm==1.2.0
python -m pip check
```

## 4. Check CUDA forward/backward execution

```bash
python tools/check_environment.py --device "$DEVICE" --check-scan \
  --json "$RESULTS_ROOT/environment_check.json"
```

Use the passing environment for training, simulation and measurement.

## 5. Optional CPU dependency check

```bash
python tools/check_environment.py --device cpu \
  --json "$RESULTS_ROOT/environment_check_cpu.json"
```

## 6. Optional FLOP analysis dependency

```bash
python -m pip install -c environment/constraints-torch113.txt -r environment/requirements-optional.txt
```

## 7. Select a physical GPU

Example: expose physical GPU 1 as logical `cuda:0`.

```bash
export CUDA_VISIBLE_DEVICES="1"
export DEVICE="cuda:0"
python tools/check_environment.py --device "$DEVICE" --check-scan \
  --json "$RESULTS_ROOT/environment_check_gpu1.json"
```
