# Validation Results

## Summary

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

## Interpretation

`float16: PASS` is the required success condition.

`float32: FAIL` is expected.

The validated path is intended for half precision attention. Full precision float32 attention is not supported by the validated route.

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

## Failure Criteria

Treat the release as not validated for a target environment if:

- `float16` fails in either validation script.
- A1111 fails to launch with `--xformers`.
- Generation produces an xFormers runtime error.
- Output contains non-finite values.
- The installed PyTorch, CUDA, xFormers, or MSLK versions do not match the release environment.
