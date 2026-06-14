import traceback
import torch

from mslk.attention.fmha import triton_splitk

# Tune down shared-memory use for RTX 5070 Laptop / sm_120 / head_dim=512.
triton_splitk.FwOp.BLOCK_N = 32
triton_splitk.FwOp.NUM_WARPS = 2
triton_splitk.FwOp.NUM_STAGES = 1
triton_splitk.FwOp.AUTOTUNE = False

from xformers.ops import memory_efficient_attention

print("Torch:", torch.__version__, torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
print("Tuned BLOCK_N:", triton_splitk.FwOp.BLOCK_N)

def run(dtype):
    print(f"\nTesting {dtype}")
    q = torch.randn((1, 9600, 1, 512), device="cuda", dtype=dtype)
    k = torch.randn((1, 9600, 1, 512), device="cuda", dtype=dtype)
    v = torch.randn((1, 9600, 1, 512), device="cuda", dtype=dtype)

    try:
        y = memory_efficient_attention(q, k, v)
        torch.cuda.synchronize()
        print("PASS", y.shape, y.dtype, "finite=", torch.isfinite(y).all().item())
        return True
    except Exception:
        print("FAIL")
        traceback.print_exc()
        return False

fp16 = run(torch.float16)
fp32 = run(torch.float32)

print("\nSummary")
print("float16:", "PASS" if fp16 else "FAIL")
print("float32:", "PASS" if fp32 else "FAIL")

if not fp16:
    raise SystemExit(1)
