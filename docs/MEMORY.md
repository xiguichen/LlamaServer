# Memory, KV cache, and context sizing

Read this when touching `KVCacheSpec.swift`, the memory budget, context
clamping, KV quantization, YaRN, or Metal pre-warm.

## The memory ceiling is measured, never assumed

`MemoryBudget.ceiling` derives the ceiling from readings of
`os_proc_available_memory()` + `task_vm_info.phys_footprint` (the syscalls live
in `ProcessMemory`; the policy stays pure).

- `os_proc_available_memory()` returns the limit **minus the caller's current
  footprint**, so the ceiling is their **sum** (capped at physical).
- `available == 0` is a **failed read**, not a zero ceiling.
- **The largest successful reading of the launch wins.** Half of physical applies
  only when *every* reading failed, and the line says
  `from fallback (every reading failed)` so an assumption is never read as a
  measurement.
- A low but *valid* reading is deliberately respected: a config that no longer
  fits is refused by the pre-check with its numbers logged, which is the
  intended trade.
- **No assumed fraction of RAM may override a successful measurement.** A
  55%-of-physical floor was removed on 2026-10-18: it sat *above* the measured
  6.44 GB limit, so the measurement never got a vote and the logged budget was a
  number no reading produced.

### What this device actually reports

Three samples of 2026-10-18 (build `698dbbf`) each sum to **exactly
6,442,450,944 B (6 GiB)**:

| Moment | available | footprint |
|---|---|---|
| at init | 6.35 GB | 90.2 MB |
| after model load | 6.29 GB | 153.8 MB |
| idle, both contexts live | 4.36 GB | 1.78 GB |

That is **exactly the 6 GiB tier the kernel bakes for this device**, an Apple
per-device constant rather than a fraction of RAM: xnu's
`memorystatus_get_default_task_active_limit` takes
`MAX(max_task_footprint_mb, memorystatus_entitled_max_task_footprint_mb)` and the
second term is a **device-tree tunable** (`entitled_max_task_pmem`,
`TUNABLE_DT_WRITEABLE`). The entitlement only *unlocks* the tier; no SDK, build
flag or app-side choice can exceed it (an independent report from the same model
measures the same 6144 MB *with* the entitlement). The only knob above it,
`increased-debugging-memory-limit`, opens the whole DRAM config and needs a
profile that sideloading cannot carry.

Disproved (do not reuse):

- "~3.3 GB un-entitled" — a *class assumption*, never a reading on this device.
- "~9 GB" — came from adding `available` to an `rss` reading; rss is not the
  ceiling operand.
- `MemoryLimitProbe`'s "base per-process limit" half (`kern.max_task_pmem`) is
  uninformative: it returns 0 even on unsandboxed macOS, and
  `kern.max_task_footprint_mb` does not exist. Use the measured ceiling half.

### What the ceiling does not cover

The model's mmap (loading 2.78 GB of weights moved the footprint by 63.6 MB) and
the Metal compute buffers (1.23 GB of `sched_reserve` lines, private storage) are
outside it. A KV budget derived from it is a **bound on what to attempt**, not a
measurement of the engine's total.

### When readings are taken

`LlamaInference.init` reads **twice** (`at init`, `after model load`), plans the
KV clamp on the larger, and logs the budget line twice (the pre-load one exists
so a refusal leaves the numbers on disk; the post-load one is authoritative).
The file-size pre-check keeps the pre-load reading — it must run before memory is
committed, so it is deliberately conservative.

The budget line prints each reading's raw `available` + `footprint` + derived
ceiling and names the winner, so the derivation is checkable from one line. The
entitlement warning keys on the winning ceiling (below half of physical = missing
uplift) or on the fallback having run. The heartbeat (`systemMemoryLog`) prints
`footprint=` **and** `available=` in the same line, so `available + footprint` is
the live ceiling check.

## Every context counts, at its own size

The MTP path creates a **second** context on the same model with the same
`n_ctx` (`LlamaInference.setupDraftContext`), and llama.cpp commits its KV up
front. So both the clamp and the fast-prefill slack charge
`KVCacheSpec.chargedKvBytesPerToken` = target bytes/token **+ draft bytes/token**.

The draft context is typed `LLAMA_CONTEXT_TYPE_MTP`, so llama.cpp builds only the
model's nextn blocks (`llama_model_n_layer_nextn`, 1 on this model). The device
log of 2026-10-18 21:04 (build `698dbbf`) measures both at 80,128 tokens:

- target cache **1,330.25 MiB**
- draft cache **166.28 MiB**
- 2,176 B/token, exactly one of the target's eight layers

History: the earlier flat "two caches" count (`KVCacheSpec.kvCacheCount`) charged
the draft 8× what it stores and clamped a viable **142,336** tokens down to
**80,128**; charging one cache would over-approve just as badly, since the draft
holds a real (if small) cache. The clamp's warning prints the split
(`17408 target + 2176 draft x1 nextn`) so the number is checkable against
llama.cpp's own `MTL0 KV buffer size` lines.

## Which cap decided the context is classified, never guessed

`KVCacheSpec.contextClamp` → `LlamaInference.contextClamp` → `ServerViewModel`.

The UI used to print "Context reduced to \(N) tokens **to fit device memory**"
for *any* reduction — including a hard ceiling the request tripped, while the KV
cache was nowhere near its limit. The message now distinguishes a model-window
cap from a memory cap (only the latter has a memory remedy). `hardMaxContext` is
262,144 (the model's own window), so asking for the full window is no longer
silently a hard-cap case.

## KV cache sizing

### Head dims come from GGUF

Read `attention.key_length` / `value_length` (suffix-matched via
`LlamaInference.ggufMetaValue`), falling back to `n_embd/n_head`. Qwen3-4B stores
128 while the derived value is 80 — deriving alone under-counts f16 KV cost by
1.6× and approved unbackable contexts.

### Hybrid models keep KV only on full-attention layers

`KVCacheSpec.kvLayerCount` reads the GGUF `<arch>.full_attention_interval` (e.g.
Qwen3.5 SSM+attention, interval 4 → 8 of 32 layers); recurrent layers have
fixed-size state. Counting all layers over-estimates KV cost interval× (observed
36,864 vs actual 9,216 B/token → a viable 131,072 context clamped to 83,968).

### Quantized KV cache

`kvCacheType` q8_0/q4_0 is allowed only when `head_dim % 32 == 0` (llama.cpp
b11557 validates per-head). Quantized `type_v` requires flash attention, so
`flash_attn_type` is forced `ENABLED` (AUTO can resolve off → constructor throw →
NULL). Ineligible models fall back to f16 with a warn log
(`KVCacheSpec.resolveCacheType`).

Measured on device: **q4_0 reaches the full 262,144 context** where q8_0 clamps
to **160,512**.

### YaRN long-context

`longContextYarn` sets `rope_scaling_type = YARN`, `yarn_orig_ctx = 32768`, and a
factor **derived from the context being built** (`KVCacheSpec.yarnFactor(forContext:)`
— the recipe's factor *is* the extension ratio: ×4 at 131,072, ×8 at 262,144,
exactly the model card's documented 128K/256K recipes; a fixed ×4 left everything
past 131,072 unscaled). It is applied to every context (target/MTP/recreate).

It lifts the `n_ctx_train` clamp; llama.cpp then re-sets
`n_ctx_train = yarn_orig_ctx / rope_freq_scale` (`llama-context.cpp`, only when
the freq scale differs from the model's trained one), so the derived factor lands
the adjusted window exactly on the built context. Still memory-clamped, and
`hardMaxContext` (262,144) is the outer bound.

## Metal pre-warm

Gated on the context size (`KVCacheSpec.needsMetalPrewarm(contextSize:)`, >= 32k),
never on the micro-batch (only 512/1024). Load logs show `metal pre-warm done`
for those contexts.
