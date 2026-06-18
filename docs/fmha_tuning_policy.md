# MSLK FMHA Tuning Policy Documentation

## Purpose

The FMHA tuning policy layer provides a controlled way to adjust Triton FMHA split-K launch parameters without repeatedly editing `triton_splitk.py`.

The main goal is to support architecture-specific compatibility fixes and low-memory experimentation while preserving upstream defaults whenever no policy or override is requested.

This was originally introduced after discovering that some Windows Blackwell laptop GPU paths with `head_dim == 512` could exceed shared memory when using `BLOCK_N = 64`. The safer compatibility value for that path is `BLOCK_N = 32`.

Instead of globally hardcoding `BLOCK_N = 32`, the tuning layer allows runtime policy selection.

## Files

```text
mslk/attention/fmha/fmha_tuning_policy.py
mslk/attention/fmha/triton_splitk.py
tests/test_fmha_tuning_policy.py
```

## Runtime Flow

```text
FwOp.apply()
  -> FwOp.get_extra_args()
    -> existing upstream heuristics choose defaults
    -> resolve_fmha_tuning()
      -> policy rules
      -> environment overrides
      -> validation
      -> optional debug logging
    -> Triton kernel launch receives resolved meta-parameters
```

Paged attention validation also resolves the effective `BLOCK_N` through the same policy layer, so validation and launch behavior use the same value.

## Environment Variables

### MSLK_FMHA_POLICY

Controls the tuning policy mode.

Valid values:

```text
default
auto
blackwell_safe
env
off
benchmark
```

Default:

```text
default
```

#### default

Preserves the existing class/default behavior unless explicit environment overrides are set.

Use this when you want normal behavior with optional manual overrides.

Example:

```powershell
$env:MSLK_FMHA_POLICY="default"
```

#### auto

Enables architecture-aware policy rules.

Current special rule:

```text
Windows + CUDA + SM120 + Kq/head_dim == 512 + non-HIP -> BLOCK_N = 32
```

Use this for normal daily Blackwell-safe operation.

Example:

```powershell
$env:MSLK_FMHA_POLICY="auto"
```

#### blackwell_safe

Forces the Blackwell-safe policy behavior when the detected platform matches the known risk case.

Current rule:

```text
Windows + CUDA + SM120 + Kq/head_dim == 512 + non-HIP -> BLOCK_N = 32
```

Use this when explicitly validating the Blackwell compatibility path.

Example:

```powershell
$env:MSLK_FMHA_POLICY="blackwell_safe"
```

#### env

Intended for explicit environment-variable testing.

Use this when manually benchmarking different values.

Example:

```powershell
$env:MSLK_FMHA_POLICY="env"
$env:MSLK_FMHA_BLOCK_N="32"
```

#### off

Disables architecture policy and ignores environment overrides.

This preserves the provided config after validation.

Use this when you want to confirm baseline behavior.

Example:

```powershell
$env:MSLK_FMHA_POLICY="off"
```

#### benchmark

Reserved for benchmark-oriented workflows.

Currently behaves like a valid policy mode that still allows environment overrides. It is useful as a semantic marker when running test matrices.

Example:

```powershell
$env:MSLK_FMHA_POLICY="benchmark"
```

## Tunable Launch Parameters

### MSLK_FMHA_BLOCK_N

Controls the key/value tile width used by the FMHA Triton kernel.

Valid values:

```text
16
32
64
128
```

Default class value:

```text
64
```

Why it matters:

`BLOCK_N` affects how much of the key/value sequence is processed per tile. Larger values may improve throughput on GPUs with enough shared memory and occupancy headroom. Smaller values reduce shared-memory pressure and may improve compatibility on memory-constrained or architecture-sensitive paths.

Known compatibility case:

```text
Windows + SM120 + head_dim 512:
  BLOCK_N=64 may exceed shared memory
  BLOCK_N=32 is safer
```

Examples:

```powershell
$env:MSLK_FMHA_BLOCK_N="32"
```

```powershell
$env:MSLK_FMHA_BLOCK_N="64"
```

Testing notes:

* `16` may be safest but slower.
* `32` is the current Blackwell-safe value.
* `64` is the restored default.
* `128` may be faster for some large-memory paths but can increase memory pressure.

### MSLK_FMHA_BLOCK_M

Controls the query tile height used by the FMHA Triton kernel.

Valid values:

```text
16
32
64
128
```

Default class value:

```text
16
```

Why it matters:

`BLOCK_M` affects how many query positions are processed per block. Increasing it can improve efficiency for larger query workloads, but it may also increase register pressure, shared-memory usage, or reduce occupancy.

Example:

```powershell
$env:MSLK_FMHA_BLOCK_M="32"
```

Testing notes:

* Start with `16`.
* Test `32` for larger query shapes.
* Avoid raising this blindly on low-VRAM systems.

### MSLK_FMHA_NUM_WARPS

Controls the number of warps used by the Triton kernel.

Valid values:

```text
1
2
4
8
```

Default class value:

```text
2
```

Why it matters:

More warps can increase parallelism, but may also increase resource usage. Fewer warps may reduce overhead or improve occupancy for smaller tiles.

Example:

```powershell
$env:MSLK_FMHA_NUM_WARPS="2"
```

Testing notes:

* `1` may help small or memory-sensitive kernels.
* `2` is the default.
* `4` can help larger workloads.
* `8` should be tested carefully.

### MSLK_FMHA_NUM_STAGES

Controls the Triton pipeline stage count.

Valid values:

```text
1
2
3
4
5
```

Default class value:

```text
1
```

Why it matters:

More stages can improve pipelining and throughput, but may increase shared-memory/register pressure. On constrained GPUs, fewer stages are usually safer.

Example:

```powershell
$env:MSLK_FMHA_NUM_STAGES="1"
```

Testing notes:

* Keep `1` as the safe baseline.
* Test higher values only with correctness and VRAM measurements.

### MSLK_FMHA_DEBUG

Enables debug logging for the resolved tuning config.

Accepted true values:

```text
1
true
yes
on
```

Example:

```powershell
$env:MSLK_FMHA_DEBUG="1"
```

Debug output includes:

```text
policy
platform
CUDA capability
GPU name
HIP status
Kq
Kkv
B
M
Mq
split_k
is_paged
use_fp8_path
BLOCK_M
BLOCK_N
num_warps
num_stages
```

Use this to confirm that the runtime policy selected the expected values.

## Recommended Usage

### Daily Blackwell-safe use

```powershell
$env:MSLK_FMHA_POLICY="auto"
```

### Force the known safe value

```powershell
$env:MSLK_FMHA_POLICY="env"
$env:MSLK_FMHA_BLOCK_N="32"
```

### Compare against default

```powershell
$env:MSLK_FMHA_POLICY="env"
$env:MSLK_FMHA_BLOCK_N="64"
```

### Enable debug output

```powershell
$env:MSLK_FMHA_POLICY="auto"
$env:MSLK_FMHA_DEBUG="1"
```

### Clear overrides in PowerShell

```powershell
Remove-Item Env:\MSLK_FMHA_POLICY -ErrorAction SilentlyContinue
Remove-Item Env:\MSLK_FMHA_BLOCK_N -ErrorAction SilentlyContinue
Remove-Item Env:\MSLK_FMHA_BLOCK_M -ErrorAction SilentlyContinue
Remove-Item Env:\MSLK_FMHA_NUM_WARPS -ErrorAction SilentlyContinue
Remove-Item Env:\MSLK_FMHA_NUM_STAGES -ErrorAction SilentlyContinue
Remove-Item Env:\MSLK_FMHA_DEBUG -ErrorAction SilentlyContinue
```

## Suggested Benchmark Matrix

Start small:

```text
BLOCK_N: 16, 32, 64, 128
BLOCK_M: 16
num_warps: 2
num_stages: 1
```

Then expand:

```text
BLOCK_N: 16, 32, 64
BLOCK_M: 16, 32
num_warps: 1, 2, 4
num_stages: 1
```

Record:

```text
GPU name
CUDA capability
driver version
PyTorch version
Triton version
xFormers/MSLK version
policy
BLOCK_M
BLOCK_N
num_warps
num_stages
shape
dtype
head_dim/Kq
paged attention yes/no
works/fails
error message
first-run JIT time
steady-state latency
peak VRAM
output correctness check
```

## Important Notes

Triton meta-parameters such as `BLOCK_N`, `BLOCK_M`, `num_warps`, and `num_stages` are selected in Python before launch, but each unique combination can produce a separate Triton JIT specialization.

That means changing these values does not require rebuilding the package, but the first run of a new combination may pay compile/JIT cost.

Always separate warmup from measured benchmark runs.

## Current Known Rule

```text
if policy in {"auto", "blackwell_safe"}:
    if Windows and CUDA capability == (12, 0) and Kq == 512 and not HIP:
        BLOCK_N = 32
```

This is intentionally narrow.

Do not generalize it to all Blackwell GPUs, all platforms, or all head dimensions without benchmark evidence.

## Safety Guidance

Recommended safe order for experimentation:

```text
1. Confirm default behavior
2. Test BLOCK_N=32
3. Test BLOCK_N=16 if memory pressure remains high
4. Test BLOCK_N=64 for performance comparison
5. Test BLOCK_N=128 only after smaller values are stable
```

Avoid changing multiple parameters at once unless running a structured benchmark matrix.

For low-memory systems, prioritize:

```text
BLOCK_N reduction
VAE tiling
attention backend fallback
batch size reduction
offloading
quantization
```

## Validation Commands

```powershell
python -m unittest tests.test_fmha_tuning_policy
```

```powershell
python -m py_compile mslk/attention/fmha/fmha_tuning_policy.py mslk/attention/fmha/triton_splitk.py tests/test_fmha_tuning_policy.py
```

## Summary

The FMHA tuning policy layer turns one hardcoded compatibility fix into a controlled runtime tuning system.

Instead of permanently changing kernel defaults for everyone, it allows:

```text
upstream defaults
targeted Blackwell-safe behavior
manual env overrides
debug visibility
paged-attention validation consistency
future benchmark automation
```

This gives us a foundation for deeper low-VRAM and architecture-specific optimization work.
