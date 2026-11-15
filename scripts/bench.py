#!/usr/bin/env python3
"""Benchmark the model loaded in LlamaServer from the device's own log lines.

The script reads `/v1/models` for the active model, then the engine's load block
to learn whether MTP is enabled, unsupported by the model, or off, and which
stats line the decode path emits (`MTP stats` or `plain stats`). Each case
starts with a unique header so the previous case's prefix cannot be reused.

The device is on the LAN, so every request bypasses HTTP proxies. The app must
be in the foreground: iOS suspends the server when the app is backgrounded.

See docs/BENCH.md for the measurement rules and docs/PERFORMANCE.md for the
log-line semantics.
"""
from __future__ import annotations

import argparse
import http.client
import json
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

DEFAULT_URL = "http://192.168.3.3:8443"
DEFAULT_CONTEXTS = "200,8000,16000,32000"

FILLER = (
    "Speculative decoding lets a small draft model propose tokens that a larger "
    "target model verifies in one batched forward pass. The target samples every "
    "emitted token, so a rejected draft costs work but never corrupts the answer. "
    "Throughput depends on acceptance, batch width, and how much of the KV cache "
    "the verify pass must attend to. "
)
INSTRUCTION = (
    "Explain how speculative decoding works and why acceptance rate matters. "
    "Write about 220 words. Use plain prose, no lists."
)

LOAD_RE = re.compile(r"loading model '([^']+)' \(requested ctx (\d+), budget (\d+) MB\)")
CREATING_RE = re.compile(
    r"creating context \((\d+) tokens, batch (\d+).*?mtp=(true|false)\)"
)
MTP_STATE_RE = re.compile(
    r"MTP (enabled|unavailable) \(two-context speculative, n_embd (\d+), "
    r"output slots (\d+), max draft (\d+), draft batch (\d+)\)"
)
KV_DETAIL_RE = re.compile(
    r"KV cache (q\d_\d|f16) \((\d+) B/token charged \[([^\]]+)\]\), "
    r"kv layers (\d+)/(\d+), YaRN long-context "
    r"(on \(x([\d.]+), orig (\d+)\)|off), head_dim K=(\d+) V=(\d+)"
)
KV_SHORT_RE = re.compile(r"KV cache (\w+)(?:, YaRN long-context (on|off))?\.")
MODEL_LOADED_RE = re.compile(r"Model loaded \(context (\d+), build ([^)]+)\)")
CONTEXT_REDUCED_RE = re.compile(
    r"Context reduced to (\d+) tokens by the device memory budget \(requested (\d+)\)"
)

GEN_RE = re.compile(
    r"generated (\d+) tokens \(prefill=([\d.]+)s ([\d.]+)t/s "
    r"\((\d+) of (\d+), (\d+) reused\), gen=([\d.]+)s ([\d.]+)t/s, "
    r"(\d+)t/min, finish=([a-z_]+), cache now (\d+)\)"
)
MTP_STATS_RE = re.compile(
    r"MTP stats: rounds=(\d+) hits=(\d+) \((\d+)%\) accepted=(\d+) "
    r"depth=([\d.]+) no-draft=(\d+) attempts=(\d+) emitted=(\d+);"
)
PLAIN_STATS_RE = re.compile(
    r"plain stats: emitted=(\d+); gpu-wait ([\d.]+)ms/step \+ "
    r"sample ([\d.]+)ms/step \+ host ([\d.]+)ms/step = wall ([\d.]+)ms/step "
    r"\((\d+) steps\)"
)
PHASE_RE = re.compile(r"([a-z-]+) ([\d.]+)ms/(?:rd|step)")

BAD_RE = re.compile(
    r"llama_decode failed|backend is in error state|"
    r"kIOGPUCommandBufferCallbackErrorInnocentVictim"
)


@dataclass
class EngineConfig:
    model: str | None = None
    build: str | None = None
    context: int | None = None
    requested_context: int | None = None
    batch: int | None = None
    kv_type: str | None = None
    bytes_per_token: int | None = None
    kv_charge: str | None = None
    yarn: str | None = None
    mtp_requested: bool | None = None
    mtp_state: str = "unknown"  # enabled | unavailable | off | unknown
    decode_path: str = "unknown"  # mtp | plain | unknown
    notes: list[str] = field(default_factory=list)


def make_opener() -> urllib.request.OpenerDirector:
    """Bypass every proxy: the device is addressed on the local network."""
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def get_text(opener, url: str, timeout: float = 30) -> str:
    with opener.open(url, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def get_json(opener, url: str, timeout: float = 15):
    with opener.open(url, timeout=timeout) as r:
        return json.loads(r.read())


def post_json(opener, url: str, body: dict, timeout: float):
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with opener.open(req, timeout=timeout) as r:
        return json.loads(r.read())


def fetch_model(opener, base: str) -> str:
    data = get_json(opener, f"{base}/v1/models")
    models = data.get("data") or []
    if not models:
        raise RuntimeError("server returned no models")
    return models[0]["id"]


def fetch_logs(opener, base: str, name: str, tail: int) -> str:
    url = f"{base}/logs?name={name}&tail={tail}&backups=1"
    return get_text(opener, url, timeout=60)


def parse_engine(inference: str, server: str = "") -> EngineConfig:
    """Read the newest model load block. Missing lines are reported, not guessed."""
    cfg = EngineConfig()
    loads = list(LOAD_RE.finditer(inference))
    if loads:
        last = loads[-1]
        cfg.model = last.group(1)
        cfg.requested_context = int(last.group(2))
        tail = inference[last.start():]
    else:
        tail = inference[-20000:]
        cfg.notes.append("no 'loading model' line in the log tail; reload the model or raise --log-tail")

    creating = list(CREATING_RE.finditer(tail))
    if creating:
        c = creating[-1]
        cfg.context = int(c.group(1))
        cfg.batch = int(c.group(2))
        cfg.mtp_requested = c.group(3) == "true"

    mtp = list(MTP_STATE_RE.finditer(tail))
    if mtp:
        m = mtp[-1]
        cfg.mtp_state = m.group(1)
    elif cfg.mtp_requested is False:
        cfg.mtp_state = "off"
    elif cfg.mtp_requested is True:
        cfg.notes.append("mtp=true but no 'MTP enabled/unavailable' line in the tail")
    else:
        cfg.notes.append("no 'creating context' line in the tail")

    kv = list(KV_DETAIL_RE.finditer(tail))
    if kv:
        k = kv[-1]
        cfg.kv_type = k.group(1)
        cfg.bytes_per_token = int(k.group(2))
        cfg.kv_charge = k.group(3)
        cfg.yarn = (
            f"x{k.group(7)} (orig {k.group(8)})" if k.group(7) else "off"
        )
    else:
        short = list(KV_SHORT_RE.finditer(server))
        if short:
            cfg.kv_type = short[-1].group(1)
            if short[-1].group(2):
                cfg.yarn = short[-1].group(2)

    loaded = list(MODEL_LOADED_RE.finditer(server))
    if loaded:
        cfg.context = int(loaded[-1].group(1))
        cfg.build = loaded[-1].group(2)
    reduced = list(CONTEXT_REDUCED_RE.finditer(server))
    if reduced:
        cfg.requested_context = int(reduced[-1].group(2))

    if cfg.mtp_state == "enabled":
        cfg.decode_path = "mtp"
    elif cfg.mtp_state in ("unavailable", "off"):
        cfg.decode_path = "plain"
    return cfg


def parse_case(inference: str, expected_path: str = "unknown") -> dict | None:
    """Parse the newest `generated …` line and the stats line before it."""
    gens = list(GEN_RE.finditer(inference))
    if not gens:
        return None
    g = gens[-1]
    rec = {
        "generated": int(g.group(1)),
        "prefill_s": float(g.group(2)),
        "prefill_tps": float(g.group(3)),
        "decoded": int(g.group(4)),
        "prompt": int(g.group(5)),
        "reused": int(g.group(6)),
        "gen_s": float(g.group(7)),
        "gen_tps": float(g.group(8)),
        "t_min": int(g.group(9)),
        "engine_finish": g.group(10),
        "cache_now": int(g.group(11)),
        "path": "unknown",
        "phases": {},
    }

    mtp_at = inference.rfind("MTP stats:", 0, g.start())
    plain_at = inference.rfind("plain stats:", 0, g.start())
    # Prefer the stats type the load block predicted, so a stale line from a
    # previous config cannot be attributed to this case. Fall back to whichever
    # line is newest when the prediction is unknown or missing.
    use_mtp = expected_path == "mtp" and mtp_at >= 0
    use_plain = expected_path == "plain" and plain_at >= 0
    if not use_mtp and not use_plain:
        use_mtp = mtp_at >= 0 and mtp_at > plain_at
        use_plain = plain_at >= 0 and plain_at > mtp_at
    if use_mtp:
        line = inference[mtp_at:].splitlines()[0]
        m = MTP_STATS_RE.search(line)
        if m:
            rec.update(
                path="mtp",
                rounds=int(m.group(1)),
                hits=int(m.group(2)),
                hit_pct=int(m.group(3)),
                accepted=int(m.group(4)),
                depth=float(m.group(5)),
                no_draft=int(m.group(6)),
                attempts=int(m.group(7)),
                emitted=int(m.group(8)),
            )
        rec["phases"] = {k: float(v) for k, v in PHASE_RE.findall(line)}
    elif use_plain:
        line = inference[plain_at:].splitlines()[0]
        m = PLAIN_STATS_RE.search(line)
        rec["path"] = "plain"
        if m:
            rec.update(
                emitted=int(m.group(1)),
                gpu_wait=float(m.group(2)),
                sample=float(m.group(3)),
                host=float(m.group(4)),
                wall=float(m.group(5)),
                steps=int(m.group(6)),
            )
        rec["phases"] = {k: float(v) for k, v in PHASE_RE.findall(line)}
    return rec


def describe_mtp(cfg: EngineConfig) -> str:
    if cfg.mtp_state == "enabled":
        return "requested + enabled -> speculative decode (MTP stats)"
    if cfg.mtp_state == "unavailable":
        return ("requested but unavailable (model has no usable NextN head, or "
                "draft setup failed) -> plain decode (plain stats)")
    if cfg.mtp_state == "off":
        return "off -> plain decode (plain stats)"
    return "unknown (no load block in the log tail) -> infer from the stats line"


def print_config(cfg: EngineConfig) -> None:
    print("engine:")
    print(f"  model    {cfg.model or 'unknown'}")
    print(f"  build    {cfg.build or 'unknown'}")
    requested = f" (requested {cfg.requested_context})" if cfg.requested_context else ""
    print(f"  context  {cfg.context if cfg.context is not None else 'unknown'}{requested}")
    print(f"  batch    {cfg.batch if cfg.batch is not None else 'unknown'}")
    kv = cfg.kv_type or "unknown"
    if cfg.bytes_per_token:
        kv += f", {cfg.bytes_per_token} B/token"
    if cfg.kv_charge:
        kv += f" [{cfg.kv_charge}]"
    print(f"  KV       {kv}")
    print(f"  YaRN     {cfg.yarn or 'unknown'}")
    print(f"  MTP      {describe_mtp(cfg)}")
    for note in cfg.notes:
        print(f"  note     {note}")
    print()


def build_content(target: int, header: str, chars_per_token: float) -> str:
    # The chat template and instructions contribute a small, roughly constant
    # number of tokens; the target is approximate and the actual prompt token
    # count is recorded from the response.
    wanted = max(0, int((target - 40) * chars_per_token))
    filler = (FILLER * (wanted // len(FILLER) + 1))[:wanted]
    return f"{header}\n\n{filler}\n\n{INSTRUCTION}"


def render_table(rows: list[dict]) -> str:
    out = [
        "| target | prompt | reused | prefill s | prefill t/s | gen s | gen t/s "
        "| path | draft-gpu | verify-gpu | gpu-wait | wall | finish |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        phases = r.get("phases") or {}
        out.append(
            "| {target} | {prompt} | {reused} | {prefill_s:.2f} | {prefill_tps:.1f} "
            "| {gen_s:.2f} | {gen_tps:.1f} | {path} | {dg} | {vg} | {gw} | {wall} "
            "| {finish} |".format(
                target=r.get("target", "?"),
                prompt=r.get("prompt") or r.get("prompt_tokens") or "?",
                reused=r.get("reused", "?"),
                prefill_s=r.get("prefill_s", 0),
                prefill_tps=r.get("prefill_tps", 0),
                gen_s=r.get("gen_s", 0),
                gen_tps=r.get("gen_tps", 0),
                path=r.get("path", "?"),
                dg=f"{phases['draft-gpu']:.1f}" if "draft-gpu" in phases else "",
                vg=f"{phases['verify-gpu']:.1f}" if "verify-gpu" in phases else "",
                gw=f"{phases['gpu-wait']:.1f}" if "gpu-wait" in phases else "",
                wall=f"{phases['wall']:.1f}" if "wall" in phases else "",
                finish=r.get("engine_finish") or r.get("finish") or "?",
            )
        )
    return "\n".join(out)


def write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def selftest() -> int:
    """Offline checks for the log parsers. No device required."""
    enabled = """
[2026-10-18 12:00:00.000] [INFO   ] loading model 'Qwopus3.5-4B-Coder-MTP-Q4_K_M.gguf' (requested ctx 262144, budget 7680 MB)
[2026-10-18 12:00:00.100] [INFO   ] KV cache q4_0 (10368 B/token charged [9216 target + 1152 draft x1 nextn]), kv layers 8/32, YaRN long-context on (x8.0, orig 32768), head_dim K=256 V=256
[2026-10-18 12:00:00.200] [INFO   ] creating context (262144 tokens, batch 1024 [fast prefill true, kv slack 0 MB], swa_window 0, swa_full on, mtp=true) for 'Qwopus3.5-4B-Coder-MTP-Q4_K_M.gguf'
[2026-10-18 12:00:00.300] [INFO   ] MTP enabled (two-context speculative, n_embd 2560, output slots 3, max draft 2, draft batch 64)
[2026-10-18 12:00:10.000] [INFO   ] MTP stats: rounds=64 hits=60 (93%) accepted=71 depth=1.11 no-draft=0 attempts=64 emitted=128; draft 10.00ms/rd + verify 70.00ms/rd + sample 1.00ms/rd (64 rounds); draft-gpu 20.00ms/rd + verify-gpu 80.00ms/rd + emit 1.00ms/rd + host 0.10ms/rd = wall 101.10ms/rd
[2026-10-18 12:00:10.001] [INFO   ] generated 128 tokens (prefill=2.00s 100.00t/s (200 of 400, 200 reused), gen=10.00s 12.80t/s, 768t/min, finish=length, cache now 528) preview='x'
"""
    unavailable = """
[2026-11-14 23:50:34.359] [INFO   ] loading model 'Qwopus3.5-9B-v3.Q4_K_M.gguf' (requested ctx 262144, budget 7680 MB)
[2026-11-14 23:50:35.272] [INFO   ] KV cache q4_0 (10368 B/token charged [9216 target + 1152 draft x1 nextn]), kv layers 8/32, YaRN long-context on (x2.0, orig 32768), head_dim K=256 V=256
[2026-11-14 23:50:35.403] [INFO   ] creating context (63232 tokens, batch 512 [fast prefill false, kv slack 0 MB], swa_window 0, swa_full on, mtp=true) for 'Qwopus3.5-9B-v3.Q4_K_M.gguf'
[2026-11-14 23:50:35.612] [INFO   ] MTP unavailable (two-context speculative, n_embd 4096, output slots 3, max draft 2, draft batch 64)
[2026-11-14 23:51:00.000] [INFO   ] plain stats: emitted=128; gpu-wait 90.00ms/step + sample 1.00ms/step + host 0.10ms/step = wall 91.10ms/step (128 steps)
[2026-11-14 23:51:00.001] [INFO   ] generated 128 tokens (prefill=1.00s 100.00t/s (100 of 100, 0 reused), gen=16.00s 8.00t/s, 480t/min, finish=length, cache now 228) preview='x'
"""
    off = """
[2026-11-01 10:00:00.000] [INFO   ] loading model 'dense.gguf' (requested ctx 32768, budget 7680 MB)
[2026-11-01 10:00:00.200] [INFO   ] creating context (32768 tokens, batch 512 [fast prefill false, kv slack 0 MB], swa_window 0, swa_full on, mtp=false) for 'dense.gguf'
[2026-11-01 10:00:20.000] [INFO   ] plain stats: emitted=64; gpu-wait 40.00ms/step + sample 0.50ms/step + host 0.10ms/step = wall 40.60ms/step (64 steps)
[2026-11-01 10:00:20.001] [INFO   ] generated 64 tokens (prefill=0.50s 128.00t/s (64 of 64, 0 reused), gen=5.00s 12.80t/s, 768t/min, finish=eog, cache now 128) preview='x'
"""
    checks = 0

    cfg = parse_engine(enabled, "")
    assert cfg.model == "Qwopus3.5-4B-Coder-MTP-Q4_K_M.gguf", cfg.model
    assert cfg.context == 262144 and cfg.batch == 1024, cfg
    assert cfg.mtp_requested is True and cfg.mtp_state == "enabled", cfg
    assert cfg.decode_path == "mtp", cfg
    assert cfg.bytes_per_token == 10368 and cfg.yarn == "x8.0 (orig 32768)", cfg
    checks += 1
    mtp = parse_case(enabled)
    assert mtp and mtp["path"] == "mtp" and mtp["prefill_tps"] == 100.0, mtp
    assert mtp["depth"] == 1.11 and mtp["phases"]["verify-gpu"] == 80.0, mtp
    checks += 1

    cfg = parse_engine(unavailable, "")
    assert cfg.model == "Qwopus3.5-9B-v3.Q4_K_M.gguf", cfg.model
    assert cfg.context == 63232 and cfg.batch == 512, cfg
    assert cfg.mtp_state == "unavailable" and cfg.decode_path == "plain", cfg
    assert cfg.yarn == "x2.0 (orig 32768)", cfg
    checks += 1
    plain = parse_case(unavailable)
    assert plain and plain["path"] == "plain" and plain["gen_tps"] == 8.0, plain
    assert plain["phases"]["gpu-wait"] == 90.0 and plain["steps"] == 128, plain
    checks += 1

    cfg = parse_engine(off, "")
    assert cfg.mtp_requested is False and cfg.mtp_state == "off", cfg
    assert cfg.decode_path == "plain", cfg
    checks += 1
    plain = parse_case(off)
    assert plain and plain["path"] == "plain" and plain["generated"] == 64, plain
    checks += 1

    server = (
        "[2026-11-14 23:50:35.641] [INFO   ] Context reduced to 63232 tokens "
        "by the device memory budget (requested 262144).\n"
        "[2026-11-14 23:50:35.641] [INFO   ] KV cache q4_0, YaRN long-context on.\n"
        "[2026-11-14 23:50:35.641] [INFO   ] Model loaded (context 63232, "
        "build llamaserver-241e47e). HTTP server listening on port 8443.\n"
    )
    cfg = parse_engine(unavailable, server)
    assert cfg.build == "llamaserver-241e47e" and cfg.context == 63232, cfg
    assert cfg.requested_context == 262144, cfg
    checks += 1

    # The newest generation wins when the log holds several.
    twice = unavailable + unavailable.replace("128 tokens", "96 tokens")
    rec = parse_case(twice)
    assert rec and rec["generated"] == 96, rec
    checks += 1

    print(f"selftest: OK ({checks} groups)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Benchmark the model loaded in LlamaServer, MTP-aware.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python3 scripts/bench.py\n"
            "  python3 scripts/bench.py --contexts 200,8000,16000,32000,48000\n"
            "  python3 scripts/bench.py --repeats 3 --label mtp-on\n"
            "  python3 scripts/bench.py --selftest\n"
        ),
    )
    ap.add_argument("--url", default=DEFAULT_URL, help=f"device base URL (default {DEFAULT_URL})")
    ap.add_argument("--contexts", default=DEFAULT_CONTEXTS,
                    help=f"comma-separated target prompt sizes (default {DEFAULT_CONTEXTS})")
    ap.add_argument("--max-tokens", type=int, default=128,
                    help="tokens to generate per case (default 128)")
    ap.add_argument("--repeats", type=int, default=1,
                    help="repeats per target, each with a unique prompt (default 1)")
    ap.add_argument("--label", default="", help="free-form tag stored in the JSON")
    ap.add_argument("--out", default="",
                    help="JSON output path (default .superpowers/bench/results/<time>-<model>.json)")
    ap.add_argument("--timeout", type=float, default=1800,
                    help="per-request timeout in seconds (default 1800)")
    ap.add_argument("--log-tail", type=int, default=12000,
                    help="inference log lines to fetch (default 12000)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the engine config and planned cases, then exit")
    ap.add_argument("--selftest", action="store_true",
                    help="run offline parser checks and exit")
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    if args.max_tokens < 1 or args.repeats < 1:
        ap.error("--max-tokens and --repeats must be >= 1")

    try:
        contexts = [int(x) for x in args.contexts.split(",") if x.strip()]
    except ValueError:
        ap.error("--contexts must be comma-separated integers")
    if not contexts or any(c < 1 for c in contexts):
        ap.error("--contexts must contain positive integers")

    base = args.url.rstrip("/")
    opener = make_opener()
    started = datetime.now(timezone.utc)

    try:
        model = fetch_model(opener, base)
    except (urllib.error.URLError, http.client.HTTPException, OSError, ValueError) as e:
        print(f"error: cannot reach {base}: {e}", file=sys.stderr)
        print(
            "  - foreground LlamaServer on the phone (iOS suspends it in the background)\n"
            "  - check the IP/port (docs/OPERATIONS.md) and that the phone is on the same Wi-Fi\n"
            "  - the script bypasses proxies; a proxy env var is not the cause",
            file=sys.stderr,
        )
        return 2

    try:
        inference = fetch_logs(opener, base, "inference", args.log_tail)
        server = fetch_logs(opener, base, "server", 6000)
    except (urllib.error.URLError, http.client.HTTPException, OSError) as e:
        print(f"error: cannot fetch logs from {base}: {e}", file=sys.stderr)
        return 2

    cfg = parse_engine(inference, server)
    print(f"device: {base}")
    print_config(cfg)

    if cfg.context:
        fitted = [c for c in contexts if c + args.max_tokens + 32 <= cfg.context]
        dropped = [c for c in contexts if c not in fitted]
        if dropped:
            print(f"note: skipping {dropped}: context {cfg.context}, "
                  f"max_tokens {args.max_tokens}\n")
        contexts = fitted
        if not contexts:
            print("error: no requested prompt size fits the loaded context",
                  file=sys.stderr)
            return 2

    if args.dry_run:
        print("plan:")
        for target in contexts:
            print(f"  {target} target prompt tokens x {args.repeats}, "
                  f"{args.max_tokens} generated tokens")
        return 0

    chars_per_token = 4.8
    cases: list[dict] = []
    payload = {
        "schema": 1,
        "label": args.label,
        "started_at": started.isoformat(),
        "base_url": base,
        "model": model,
        "engine": asdict(cfg),
        "plan": {
            "contexts": contexts,
            "max_tokens": args.max_tokens,
            "repeats": args.repeats,
        },
        "cases": cases,
        "valid": True,
        "error": None,
    }
    out = args.out
    if not out:
        stamp = started.strftime("%Y%m%d-%H%M%S")
        slug = re.sub(r"[^A-Za-z0-9._-]+", "-", model)[:60]
        out = os.path.join(".superpowers", "bench", "results", f"{stamp}-{slug}.json")
    print(f"results: {out}\n")

    total = len(contexts) * args.repeats
    index = 0
    for repeat in range(args.repeats):
        for target in contexts:
            index += 1
            header = f"[[bench {started.strftime('%H%M%S')} case {target} rep {repeat} idx {index}]]"
            content = build_content(target, header, chars_per_token)
            body = {
                "model": model,
                "messages": [{"role": "user", "content": content}],
                "temperature": 0,
                "max_tokens": args.max_tokens,
                "stream": False,
            }
            print(f"[{index}/{total}] target={target} rep={repeat} "
                  f"chars={len(content)} ...", flush=True)
            t0 = time.time()
            try:
                data = post_json(opener, f"{base}/v1/chat/completions", body, args.timeout)
            except Exception as e:  # noqa: BLE001 - report the transport failure
                payload["valid"] = False
                payload["error"] = f"request failed: {e}"
                payload["finished_at"] = datetime.now(timezone.utc).isoformat()
                write_json(out, payload)
                print(f"error: request failed: {e}", file=sys.stderr)
                print(
                    "  the app was probably suspended or the request exceeded --timeout;\n"
                    "  foreground the app and re-run. Partial results were saved.",
                    file=sys.stderr,
                )
                return 2

            wall = time.time() - t0
            usage = data.get("usage") or {}
            choice = (data.get("choices") or [{}])[0]
            text = ((choice.get("message") or {}).get("content") or "")
            fingerprint = data.get("system_fingerprint")
            if payload.get("fingerprint") is None and fingerprint:
                payload["fingerprint"] = fingerprint
            elif fingerprint and payload.get("fingerprint") != fingerprint:
                payload["valid"] = False
                payload["error"] = (
                    f"build changed mid-run: {payload.get('fingerprint')} -> {fingerprint}"
                )
                write_json(out, payload)
                print(f"error: {payload['error']}", file=sys.stderr)
                return 3
            if data.get("model") and data["model"] != model:
                payload["valid"] = False
                payload["error"] = f"model changed mid-run: {model} -> {data['model']}"
                write_json(out, payload)
                print(f"error: {payload['error']}", file=sys.stderr)
                return 3
            # FileLogger buffers info lines for up to 1 s; GET /logs also flushes.
            time.sleep(1.2)
            try:
                tail = fetch_logs(opener, base, "inference", args.log_tail)
            except (urllib.error.URLError, http.client.HTTPException, OSError) as e:
                tail = ""
                print(f"warning: could not fetch the log after the case: {e}",
                      file=sys.stderr)

            parsed = parse_case(tail, cfg.decode_path) if tail else None
            creating = list(CREATING_RE.finditer(tail)) if tail else []
            if creating and cfg.context and int(creating[-1].group(1)) != cfg.context:
                live_ctx = int(creating[-1].group(1))
                payload["valid"] = False
                payload["error"] = f"context changed mid-run: {cfg.context} -> {live_ctx}"
                write_json(out, payload)
                print(f"error: {payload['error']}", file=sys.stderr)
                return 3
            rec = {
                "target": target,
                "repeat": repeat,
                "wall_s": round(wall, 2),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "finish": choice.get("finish_reason"),
                "fingerprint": fingerprint,
                "chars": len(content),
                "content": text[:200],
            }
            if parsed:
                rec.update(parsed)
                if rec.get("path") == "unknown":
                    print("warning: no 'MTP stats'/'plain stats' line before the "
                          "generated line; phase columns will be blank", file=sys.stderr)
                # The engine's gen figure starts after the first sampled token;
                # wall_s above includes the whole HTTP round trip.
                if rec.get("prompt") and rec["prompt"] > 500:
                    chars_per_token = len(content) / rec["prompt"]
            else:
                rec["error"] = (
                    "no 'generated …' line in the log tail; raise --log-tail, "
                    "or check that the request and the log fetch both completed"
                )
                print(f"error: {rec['error']}", file=sys.stderr)
                if BAD_RE.search(tail):
                    rec["gpu_reset"] = True
                cases.append(rec)
                payload["valid"] = False
                payload["error"] = rec["error"]
                write_json(out, payload)
                return 3

            cases.append(rec)
            write_json(out, payload)

            if rec.get("gen_tps") is not None:
                print(f"  prompt={rec['prompt']} tok  prefill={rec['prefill_s']:.2f}s "
                      f"{rec['prefill_tps']:.1f} t/s  gen={rec['gen_tps']:.2f} t/s  "
                      f"path={rec['path']}  finish={rec['engine_finish']}", flush=True)

            # A GPU reset invalidates the run even if a later decode recovered.
            # Only the recent window is checked so an old fault cannot abort a
            # healthy run.
            if BAD_RE.search("\n".join(tail.splitlines()[-200:])):
                rec["gpu_reset"] = True
                payload["valid"] = False
                payload["error"] = (
                    "GPU reset in the recent log window; this is not a measurement"
                )
                write_json(out, payload)
                print(f"error: {payload['error']}", file=sys.stderr)
                return 3

    payload["finished_at"] = datetime.now(timezone.utc).isoformat()
    write_json(out, payload)

    print()
    print(render_table(cases))
    if args.repeats > 1:
        print("\nmedians:")
        for target in contexts:
            group = [r for r in cases if r["target"] == target and r.get("gen_tps")]
            if group:
                print(f"  target={target}: prefill "
                      f"{statistics.median(r['prefill_tps'] for r in group):.1f} t/s, gen "
                      f"{statistics.median(r['gen_tps'] for r in group):.1f} t/s "
                      f"(n={len(group)})")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
