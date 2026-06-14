# Patch Notes

## triton_splitk.py

Original:

```python
BLOCK_N: int = 64
```

Modified:

```python
# Lowered for Windows Blackwell laptop GPUs where head_dim=512 exceeds shared memory with 64.
BLOCK_N: int = 32
```

## Reason

The tested Windows Blackwell laptop environment encountered shared-memory pressure with the original block size for the validated head_dim=512 attention workload.

Lowering BLOCK_N to 32 allowed the validated half precision attention path to execute successfully through the companion xFormers build.
