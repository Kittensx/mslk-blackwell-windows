import traceback
import torch
from xformers.ops import memory_efficient_attention

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

print("Torch:", torch.__version__, torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))

fp16 = run(torch.float16)
fp32 = run(torch.float32)

print("\nSummary")
print("float16:", "PASS" if fp16 else "FAIL")
print("float32:", "PASS" if fp32 else "FAIL")

if not fp16:
    raise SystemExit(1)
