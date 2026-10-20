#!/usr/bin/env python3
"""Measure MTP generation speed at several prompt sizes against a running LlamaServer.

Parses the engine's own `generated …` line (prefill=, gen=) plus the `MTP stats:`
line (accepted, depth, draft/verify GPU phases). Each case has a unique prompt
header so the previous case's prefix cannot be reused.
"""
import argparse, json, re, sys, time, urllib.request

FILLER = ("Speculative decoding lets a small draft model propose tokens that a larger "
          "target model verifies in one batched forward pass. The target samples every "
          "emitted token, so a rejected draft costs work but never corrupts the answer. "
          "Throughput depends on acceptance, batch width, and how much of the KV cache "
          "the verify pass must attend to. ") * 400

INSTRUCTION = ("Explain how speculative decoding works and why acceptance rate matters. "
               "Write about 220 words. Use plain prose, no lists.")

GEN_RE = re.compile(r"generated (\d+) tokens \(prefill=([\d.]+)s ([\d.]+)t/s "
                    r"\((\d+) of (\d+), (\d+) reused\), gen=([\d.]+)s ([\d.]+)t/s, "
                    r"(\d+)t/min, finish=([a-z_]+), cache now (\d+)\)")
MTP_RE = re.compile(r"MTP stats: rounds=(\d+) hits=(\d+) \((\d+)%\) accepted=(\d+) "
                    r"depth=([\d.]+) no-draft=(\d+) attempts=(\d+) emitted=(\d+);(.*)")
LOAD_RE = re.compile(r"MTP enabled \(.*output slots (\d+), max draft (\d+)\)")
PHASE_RE = re.compile(r"([a-z-]+) (\d+\.\d+)ms/rd")

def post(url, body, timeout=1800):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read())
    return data, time.time() - t0

def get(url, timeout=60):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")

def log(url, name, tail):
    try:
        return get(f"{url}/logs?name={name}&tail={tail}")
    except Exception as e:
        return f"<<log fetch failed: {e}>>"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--contexts", default="200,9000,18000,27000")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    url = a.url.rstrip("/")

    model = json.loads(get(f"{url}/v1/models"))["data"][0]["id"]
    load = log(url, "inference", 4000)
    m = LOAD_RE.search(load)
    engine = {"model": model,
              "output_slots": int(m.group(1)) if m else None,
              "max_draft": int(m.group(2)) if m else None}
    print(f"model={model} max_draft={engine['max_draft']} output_slots={engine['output_slots']}")
    if engine["max_draft"] is None:
        print("WARNING: no 'MTP enabled' line in the log tail — is MTP on?")

    results = []
    for target in [int(x) for x in a.contexts.split(",") if x.strip()]:
        header = f"[[bench case {target}]]"
        body_text = header + "\n\n" + FILLER[:(max(0, target - 300) * 4)] + "\n\n" + INSTRUCTION
        body = {"model": model, "messages": [{"role": "user", "content": body_text}],
                "temperature": 0, "max_tokens": a.max_tokens, "stream": False}
        data, wall = post(f"{url}/v1/chat/completions", body)
        usage = data.get("usage") or {}
        choice = (data.get("choices") or [{}])[0]
        text = ((choice.get("message") or {}).get("content") or "")
        time.sleep(1.5)
        tail = log(url, "inference", 300)
        gens = list(GEN_RE.finditer(tail))
        mtps = list(MTP_RE.finditer(tail))
        g = gens[-1] if gens else None
        mtp = mtps[-1] if mtps else None
        rec = {"target": target, "wall_s": round(wall, 2),
               "prompt_tokens": usage.get("prompt_tokens"),
               "completion_tokens": usage.get("completion_tokens"),
               "finish": choice.get("finish_reason"), "text_chars": len(text)}
        if g:
            rec.update(prefill_s=float(g.group(2)), prefill_tps=float(g.group(3)),
                       decoded=g.group(4), prompt=g.group(5), reused=g.group(6),
                       gen_s=float(g.group(7)), gen_tps=float(g.group(8)),
                       engine_finish=g.group(10))
        if mtp:
            rec.update(rounds=int(mtp.group(1)), hits_pct=int(mtp.group(3)),
                       accepted=int(mtp.group(4)), depth=float(mtp.group(5)),
                       no_draft=int(mtp.group(6)), mtp_emitted=int(mtp.group(8)),
                       phases={k: float(v) for k, v in PHASE_RE.findall(mtp.group(9))})
        results.append(rec)
        print(f"prompt={rec.get('prompt_tokens')} tok  prefill={rec.get('prefill_tps')} t/s  "
              f"gen={rec.get('gen_tps')} t/s  accepted/round={rec.get('depth')}  "
              f"emitted={rec.get('completion_tokens')}  finish={rec.get('engine_finish')}")
        with open(a.out, "w") as f:
            json.dump({"engine": engine, "results": results}, f, indent=2)
    print(f"\nwrote {a.out}")

if __name__ == "__main__":
    main()
