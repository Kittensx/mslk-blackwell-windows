# Release Notes: v0.1.0-blackwell-sm120

Initial experimental release for Windows Blackwell / SM120 MSLK validation.

## Release Asset

Upload this file as a GitHub Release asset:

```text
mslk-1.1.0.post1+sm120a1111-py3-none-any.whl
```

## Tested Environment

```text
Windows x64
GPU: NVIDIA GeForce RTX 5070 Laptop GPU
Compute Capability: 12.0 / SM120
Python: 3.10.6
PyTorch: 2.11.0+cu128
CUDA Runtime: 12.8
CUDA Toolkit: 12.8.61
MSLK: 1.1.0.post1+sm120a1111
Primary validation app: AUTOMATIC1111 Stable Diffusion WebUI
Companion validation: xFormers 0.0.35+af84367e
```

## SHA256 Checksum

```text
682cd2c4b85c6ef192ff9f1009c3b2af079bdcd006c54e4348c33da0c9763e58  mslk-1.1.0.post1+sm120a1111-py3-none-any.whl
```

Verify on Windows with:

```bat
certutil -hashfile mslk-1.1.0.post1+sm120a1111-py3-none-any.whl SHA256
```

## Actual MSLK Change

File:

```text
mslk/attention/fmha/triton_splitk.py
```

Original:

```python
BLOCK_N: int = 64
```

Modified:

```python
# Lowered for Windows Blackwell laptop GPUs where head_dim=512 exceeds shared memory with 64.
BLOCK_N: int = 32
```

This lowers Triton Split-K shared-memory pressure for the validated RTX 5070 Laptop / SM120 / head_dim=512 case.

## Installation

```bat
python -m pip uninstall -y mslk
python -m pip install --no-deps mslk-1.1.0.post1+sm120a1111-py3-none-any.whl
```

For use with the companion xFormers wheel:

```bat
python -m pip uninstall -y xformers mslk
python -m pip install --no-deps mslk-1.1.0.post1+sm120a1111-py3-none-any.whl
python -m pip install --no-deps xformers-0.0.35+af84367e.d20260613-py39-none-win_amd64.whl
```

## Validation Results

```text
Torch: 2.11.0+cu128
CUDA: 12.8
GPU: NVIDIA GeForce RTX 5070 Laptop GPU
Compute Capability: (12, 0)

Testing torch.float16
PASS torch.Size([1, 9600, 1, 512]) torch.float16 finite=True

Testing torch.float32
FAIL

Summary:
float16: PASS
float32: FAIL
```

`float16: PASS` is the required success condition. `float32: FAIL` is expected. Full precision float32 attention is not supported by the validated route.

## A1111 Runtime Validation

```text
AUTOMATIC1111 was launched with --xformers.
WebUI launched successfully.
Model loaded successfully.
Generation completed successfully.
No xFormers runtime errors were observed.
Fast iteration testing completed.
```

This was validation of the full MSLK + xFormers stack, not MSLK alone.

## Release Classification

Mark this GitHub Release as a pre-release.
