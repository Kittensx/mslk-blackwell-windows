# Changelog

## v0.1.0-blackwell-sm120

Initial experimental release.

- Added Windows-compatible MSLK wheel for Blackwell / SM120 validation.
- Applied one-line Triton Split-K BLOCK_N adjustment from 64 to 32.
- Validated with RTX 5070 Laptop GPU.
- Validated with PyTorch 2.11.0+cu128 and CUDA 12.8.
- Added validation scripts for xFormers/A1111 attention shape.
- Documented expected float16 success and float32 failure.
- Added SHA256 checksum.
