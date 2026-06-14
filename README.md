# mslk-blackwell-windows

Experimental Windows MSLK build for NVIDIA Blackwell (SM120) GPUs. Includes a validated Triton Split-K adjustment for head_dim=512 workloads used by xFormers.

## 1. Project Overview

This repository publishes documentation, validation scripts, checksums, and release metadata for an experimental Windows-compatible MSLK build validated on NVIDIA Blackwell laptop hardware.

The first release target is:

```text
mslk-1.1.0.post1+sm120a1111-py3-none-any.whl
```

The wheel is not stored directly in this repository. It should be uploaded as a GitHub Release asset.

## 2. Experimental Warning

This is an experimental compatibility build. It is not official upstream MSLK Windows support and should not be treated as a general-purpose MSLK distribution.

Use it only if you understand the risks of custom GPU runtime dependencies, including unsupported kernel behavior, version mismatch, numerical instability, and app-specific validation gaps.

## 3. Tested Environment

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

## 4. Why This Build Exists

Upstream MSLK does not currently provide an official Windows-supported wheel for this use case.

This repository provides a Windows-compatible MSLK build validated on NVIDIA Blackwell SM120 hardware. The build was needed to support a companion xFormers Blackwell Windows build, and validation was performed on RTX 5070 Laptop GPU hardware.

This repository does not imply official upstream support.

## 5. Actual MSLK Change

The functional source change is intentionally minimal.

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

This lowers the Triton Split-K block size. It reduces shared-memory pressure for the validated RTX 5070 Laptop / SM120 case. The tested workload involved `head_dim=512`.

The change is intentionally narrow and should be treated as experimental.

## 6. Release Artifact

GitHub Release asset:

```text
mslk-1.1.0.post1+sm120a1111-py3-none-any.whl
```

Do not commit wheel files into this repository. Release binaries should be attached to the GitHub Release.

## 7. SHA256 Verification

Known SHA256 checksum:

```text
682cd2c4b85c6ef192ff9f1009c3b2af079bdcd006c54e4348c33da0c9763e58  mslk-1.1.0.post1+sm120a1111-py3-none-any.whl
```

Verify on Windows with:

```bat
certutil -hashfile mslk-1.1.0.post1+sm120a1111-py3-none-any.whl SHA256
```

The printed hash must match the value above.

## 8. Installation

Install MSLK alone:

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

## 9. Validation

MSLK was validated through the xFormers memory-efficient attention path used by A1111.

Run validation from the target virtual environment:

```bat
python tests\test_xformers_a1111_shape.py
python tests\test_xformers_a1111_tuned.py
```

The required success condition is `float16: PASS`. The expected full-precision result is `float32: FAIL`.

## 10. Validation Scripts

This repository includes:

```text
tests/test_xformers_a1111_shape.py
tests/test_xformers_a1111_tuned.py
```

These scripts are included because MSLK was validated through the companion xFormers attention path. The tuned script applies the known Split-K settings used for the RTX 5070 Laptop / SM120 / head_dim=512 validation case.

## 11. Actual Validation Results

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

`float16: PASS` is the required success condition. `float32: FAIL` is expected. The validated path is intended for half precision attention. Full precision float32 attention is not supported by the validated route.

A1111 runtime validation:

```text
AUTOMATIC1111 was launched with --xformers.
WebUI launched successfully.
Model loaded successfully.
Generation completed successfully.
No xFormers runtime errors were observed.
Fast iteration testing completed.
```

This was validation of the full MSLK + xFormers stack, not MSLK alone.

## 12. Relationship To xFormers

MSLK is a required companion dependency for the validated xFormers Blackwell Windows build.

This repository publishes the MSLK side separately because users may search specifically for Windows MSLK support.

Companion repository:

```text
xformers-blackwell-windows
https://github.com/YOUR_USERNAME/xformers-blackwell-windows
```

## 13. Recommended Usage

Install this MSLK wheel before installing the companion xFormers Blackwell wheel.

Recommended install order:

```text
1. mslk
2. xformers
3. application validation
```

## 14. Limitations

- This is not official upstream MSLK Windows support.
- RTX 5070 Laptop GPU only validated.
- Windows only validated.
- Float32 attention unsupported in the validated path.
- No Linux testing.
- No multi-GPU testing.
- No training validation.
- No guarantee for all RTX 50-series cards.
- No guarantee for all xFormers workloads.

## 15. Future Work

- RTX 5060 validation.
- RTX 5070 desktop validation.
- RTX 5080 validation.
- RTX 5090 validation.
- Additional CUDA versions.
- Additional PyTorch versions.
- Upstream Windows compatibility investigation.
- Performance benchmarking.
- VRAM benchmarking.

## 16. Credits

Credit belongs to the upstream MSLK, PyTorch, Triton, CUDA, and xFormers maintainers. This repository documents an experimental Windows packaging and validation path for a specific Blackwell laptop environment.

## 17. License Notes

This repository distributes documentation, validation scripts, and release metadata for an experimental Windows MSLK build. It does not relicense upstream MSLK or xFormers projects. See `LICENSE_NOTES.md`.
