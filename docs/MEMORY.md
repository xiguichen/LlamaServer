# Memory, KV cache, and context sizing

Read this when touching `KVCacheSpec.swift`, the memory budget, context
clamping, KV quantization, YaRN, or Metal pre-warm.

## The memory ceiling is measured, never assumed

`MemoryBudget.settledCeiling` derives the planning ceiling from readings of
`os_proc_available_memory()` + `task_vm_info.phys_footprint` (the syscalls live
in `ProcessMemory`; the policy stays pure). `MemoryBudget.ceiling` is the
conservative smallest-wins fallback it uses when no plateau exists.

- `os_proc_available_memory()` returns the limit **minus the caller's current
  footprint**, so the ceiling is their **sum** (capped at physical).
- `available == 0` is a **failed read**, not a zero ceiling.
- **While the tier is unchanged, readings measure the same constant.** The
  ceiling is a property of the device+state, not something that rises as the
  process settles. iOS 27 adds one deliberate exception: Game Mode raises the
  12GB A19 Pro tier from 6GiB to 8GiB mid-launch (see below).
- **Within a state, when readings disagree, the smallest successful reading
  wins.** A disagreement means one operand was stale, and the failure modes are
  asymmetric: planning above the real ceiling is a jetsam SIGKILL, planning
  below it only costs context tokens. A failed reading never wins. A *rise* is
  accepted only as a **corroborated plateau** (three consecutive readings at the
  same new limit); a lone high reading still loses.
- Half of physical applies only when *every* reading failed, and the line says
  `from fallback (every reading failed)` so an assumption is never read as a
  measurement.
- A low but *valid* reading is deliberately respected: a config that no longer
  fits is refused by the pre-check with its numbers logged, which is the
  intended trade.
- **No assumed fraction of RAM may override a successful measurement.** A
  55%-of-physical floor was removed on 2026-10-18: it sat *above* the measured
  6.44 GB limit, so the measurement never got a vote and the logged budget was a
  number no reading produced.
- **No stale reading may inflate the budget either.** The old "largest reading
  wins" rule let `available` paired with a 4.58 GB **rss** value (not
  `phys_footprint`) produce a ~9.17 GB ceiling, which approved a 262,144-token
  q8_0 KV the device cannot hold. Resolving downward is what makes the 6 GiB
  tier a hard limit rather than a suggestion. The later Game Mode uplift is
  accepted only as a corroborated plateau, so the same guard still rejects a
  single stale reading.

### What this device actually reports

Three samples of 2026-10-18 (build `698dbbf`) each sum to **exactly
6,442,450,944 B (6 GiB)**:

| Moment | available | footprint |
|---|---|---|
| at init | 6.35 GB | 90.2 MB |
| after model load | 6.29 GB | 153.8 MB |
| idle, both contexts live | 4.36 GB | 1.78 GB |

That is **the iOS 26 6 GiB tier the kernel bakes for this device**, an Apple
per-device constant rather than a fraction of RAM: xnu's
`memorystatus_get_default_task_active_limit` takes
`MAX(max_task_footprint_mb, memorystatus_entitled_max_task_footprint_mb)` and the
second term is a **device-tree tunable** (`entitled_max_task_pmem`,
`TUNABLE_DT_WRITEABLE`). The entitlement only *unlocks* the tier; on iOS 26 no
SDK, build flag or app-side choice exceeds it (an independent report from the
same model measures the same 6144 MB *with* the entitlement). The next knob,
`increased-debugging-memory-limit`, opens the whole DRAM config and needs a
profile that sideloading cannot carry. iOS 27's Game Mode tier (below) is the
supported way past 6 GiB without that profile.

### iOS 27 Game Mode raises the tier to 8 GiB

iOS 27 raises the 12 GB A19 Pro per-process ceiling from **6 GiB (6,144 MiB)
to 8 GiB (8,192 MiB) while Game Mode is active**. This is not a larger
entitlement: `increased-memory-limit` still unlocks only the 6 GiB tier, and
the extra 2 GiB appears only when the OS turns Game Mode on for the app. Game
Mode activates a few seconds after a Game-Mode-capable app is foregrounded, so
the same launch can read 6,144 MiB first and 8,192 MiB moments later. Sources:
the LiveContainer iOS 27 report (8 GB requires Game Mode; add
`GCSupportsGameMode`), a Playport device trace (6,144 MiB at app start, 8,192
MiB once Game Mode's banner appeared), and the iOS 27 beta 3 report of the
raised cap on 12 GB phones.

The app opts in through `LSSupportsGameMode` (iOS 26+), `GCSupportsGameMode`
(iOS 18+) and `LSApplicationCategoryType: public.app-category.games` in
`project.yml`. Consequences:

- the system shows the Game Mode banner while it is on;
- App Review rejects non-games that opt in, so this is for the sideloaded build;
- if the user disables Game Mode for the app, the ceiling drops back to 6 GiB
  and the budget follows the measurement.

Because the tier moves mid-launch, the planning reading is a **settled plateau**,
not a single sample: `ProcessMemory.settleCeiling` samples every 250 ms after
the model load (up to 5 s) and stops when three consecutive readings agree.
`MemoryBudget.settledCeiling` accepts the highest *corroborated* plateau and
still falls back to smallest-wins when no plateau exists, so a lone high reading
(the old `available + rss` 9.17 GB shape) never wins.

**Measured 2026-11-07 (build `85e00fc`, iOS 27, Game Mode on):** the init
reading was already `8,499,427,416 B + 90,507,176 B = 8,589,934,592 B`
(**8 GiB**); the post-load plateau settled at `8,412,803,144 B + 177,131,448 B
= 8,589,934,592 B` after 0.51 s. The budget became **8,053,063,680 B
(7.5 GiB)**, up from 5,905,580,032 B (5.91 GB) on the 6 GiB tier, with no
jetsam or error lines. q4_0 still reaches the model's full 262,144-token
window. The extra budget is available to the KV clamp, but q8_0's new ceiling
has **not** been re-measured yet (it clamped to 160,512 tokens on the 6 GiB
tier); run it before quoting a q8_0 context.

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

`LlamaInference.init` takes a conservative reading at init (for the file-size
pre-check and so a refusal leaves the numbers on disk), then after the model
load samples the ceiling until it is stable (`ProcessMemory.settleCeiling`:
three agreeing readings 250 ms apart) and plans the KV clamp on
`MemoryBudget.settledCeiling`. The log line lists every sample and names the
settled winner; a 6 GiB init followed by an 8 GiB settled plateau is therefore
visible in one line. The probe line (`MemoryLimitProbe`) is logged at both
points, so a reader can see which ceiling the context was actually planned
against. If the sampler never sees a plateau, `settledCeiling` falls back to the
conservative smallest-successful rule.

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
