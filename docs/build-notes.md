# Build Notes

## Purpose

These notes document the release context for the experimental Windows Blackwell MSLK package:

```text
mslk-1.1.0.post1+sm120a1111-py3-none-any.whl
```

The package is intended to be distributed as a GitHub Release asset, not committed to the repository.

## Target Environment

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

## Build Strategy

This release follows a narrow compatibility posture:

1. Keep the functional source change minimal.
2. Publish MSLK as a companion dependency for the xFormers Blackwell Windows build.
3. Keep wheel binaries outside the git-tracked repository.
4. Use GitHub Release assets for binary distribution.
5. Validate through the A1111/xFormers attention path before publishing.

## Important Compatibility Notes

- The validated path is half precision.
- The observed float32 failure is expected and documented.
- The MSLK wheel was validated through the full MSLK + xFormers stack.
- If xFormers is unstable, use PyTorch scaled dot-product attention as the stable fallback.

## Release Asset Policy

Do not commit:

```text
*.whl
*.zip
*.tar.gz
```

Upload this file to the GitHub Release instead:

```text
mslk-1.1.0.post1+sm120a1111-py3-none-any.whl
```
