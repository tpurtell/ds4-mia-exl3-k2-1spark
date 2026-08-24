#!/usr/bin/env python3
"""Reproduce issue #27 starvation on the LIVE dspark server.

8K x N cold wave (N = max_num_seqs) using the EXACT protocol from issue #27:
  - unique front-entropy so prefixes don't collide in cache
  - max_tokens=min_tokens=256, ignore_eos=true, temperature=0, seed per lane
  - streaming + usage; per-lane TTFT and decode tok/s
Reports per-lane TTFT/decode + before/after counter deltas (preemptions, MTP,
prefix hits). Streaming-only /v1/completions over raw HTTP.
"""
import json, time, urllib.request, urllib.error, sys, threading, statistics

API = "http://127.0.0.1:8888"
MODEL = "deepseek-v4-flash-0731"
N_LANES = int(sys.argv[1]) if len(sys.argv) > 1 else 4
TARGET_IN = int(sys.argv[2]) if len(sys.argv) > 2 else 8192
OUT = 256
SEED_BASE = 27_000


def post(url, body, timeout=600):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    return urllib.request.urlopen(req, timeout=timeout)


def tokenize(text):
    with post(f"{API}/tokenize", {"model": MODEL, "prompt": text}) as r:
        return json.load(r)["count"]


def build_unique_prompt(target, nonce):
    # Unique front entropy so no two lanes share a prefix in cache.
    front = (f"issue-27 lane-nonce-{nonce} " +
             "ABCDEFGHJKLMNPQRSTUVWXYZ0123456789" * 64)
    unit = "benchmark context datum alpha beta gamma delta epsilon "
    text = front + "\n\n" + unit * (target // 3)
    while True:
        c = tokenize(text)
        if c >= target:
            return text, c
        text += unit * max(1, (target - c) // 3)


def run_lane(idx, prompt, prompt_len, out):
    res = {"idx": idx, "prompt_tokens": prompt_len,
           "ttft": None, "first_byte": None, "done": None,
           "out_tokens": 0, "error": None, "usage": None}
    body = {
        "model": MODEL, "prompt": prompt,
        "max_tokens": OUT, "min_tokens": OUT,
        "ignore_eos": True, "temperature": 0.0,
        "seed": SEED_BASE + idx, "stream": True,
        "stream_options": {"include_usage": True},
    }
    t_send = time.perf_counter()
    try:
        with post(f"{API}/v1/completions", body, timeout=900) as resp:
            buf = b""
            for chunk in iter(lambda: resp.read(4096), b""):
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line = line.strip()
                    if not line.startswith(b"data:"):
                        continue
                    data = line[5:].strip()
                    if data == b"[DONE]":
                        continue
                    try:
                        ev = json.loads(data)
                    except Exception:
                        continue
                    chs = ev.get("choices") or []
                    if chs and chs[0].get("text"):
                        if res["first_byte"] is None:
                            res["first_byte"] = time.perf_counter()
                            res["ttft"] = res["first_byte"] - t_send
                        res["out_tokens"] += 1
                    if ev.get("usage"):
                        res["usage"] = ev["usage"]
    except Exception as e:
        res["error"] = repr(e)
    res["done"] = time.perf_counter()
    res["wall"] = res["done"] - t_send
    # Streaming text deltas batch MTP-accepted tokens, so event-counting
    # (res["out_tokens"]) is unreliable. Derive decode rate from the
    # authoritative OUT-token window (server always emits exactly OUT tokens:
    # min=max=OUT, ignore_eos=true; verified by generation_tokens_total delta).
    if res["first_byte"]:
        res["decode_toks"] = OUT
        res["decode_secs"] = res["done"] - res["first_byte"]
        res["decode_tps"] = res["decode_toks"] / res["decode_secs"] if res["decode_secs"] > 0 else 0.0
        res["mean_itl_ms"] = (res["decode_secs"] / res["decode_toks"]) * 1000.0 if res["decode_toks"] > 0 else 0.0
    return res


def metric(name, m):
    for line in m.splitlines():
        if line.startswith(name + "{") and not "_created" in line:
            i = line.rfind("}")
            if i >= 0:
                return float(line[i + 2:])
    return None


def main():
    print(f"[repro] N={N_LANES} target_in={TARGET_IN} out={OUT}")
    m_before = urllib.request.urlopen(API + "/metrics").read().decode()
    s0 = time.perf_counter()
    prompts = [build_unique_prompt(TARGET_IN, f"lane{idx}-{int(time.time())}")
               for idx in range(N_LANES)]
    for i, (t, c) in enumerate(prompts):
        print(f"  lane{i} prompt tokens = {c}")
    wave_s = time.perf_counter()
    threads = []
    results = [None] * N_LANES

    def worker(i):
        results[i] = run_lane(i, prompts[i][0], prompts[i][1], OUT)

    for i in range(N_LANES):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t); t.start()
    for t in threads:
        t.join()
    wave_e = time.perf_counter()
    print(f"\n[repro] wave wall = {wave_e - wave_s:.3f} s\n")
    print(f"{'lane':>4} {'in':>7} {'out':>5} {'TTFT(s)':>9} {'dec_tps':>8} {'meanITL(ms)':>11} {'wall(s)':>8}")
    tps = []
    for r in results:
        if r is None:
            print("  <no result>"); continue
        if r.get("error"):
            print(f"  lane{r['idx']} ERROR {r['error']}"); continue
        tps.append(r["decode_tps"])
        print(f"{r['idx']:>4} {r['prompt_tokens']:>7} {r['out_tokens']:>5} "
              f"{(r['ttft'] or 0):>9.3f} {r.get('decode_tps',0):>8.2f} "
              f"{r.get('mean_itl_ms',0):>11.1f} {r['wall']:>8.2f}")
    if tps:
        print(f"\n  decode min/median/max = {min(tps):.2f} / "
              f"{statistics.median(tps):.2f} / {max(tps):.2f} tok/s")
        print(f"  spread (max/min)     = {max(tps)/min(tps):.2f}x")
    time.sleep(2)
    m_after = urllib.request.urlopen(API + "/metrics").read().decode()

    def delta(nm):
        return (metric(nm, m_after) or 0) - (metric(nm, m_before) or 0)

    print("\n[counter deltas over wave]")
    print(f"  preemptions             = {delta('vllm:num_preemptions_total'):+.0f}")
    print(f"  prompt_tokens_total     = {delta('vllm:prompt_tokens_total'):+.0f}")
    print(f"  generation_tokens_total = {delta('vllm:generation_tokens_total'):+.0f}")
    print(f"  prefix_cache_hits_total = {delta('vllm:prefix_cache_hits_total'):+.0f}")
    d = delta("vllm:spec_decode_num_draft_tokens_total") or 0
    a = delta("vllm:spec_decode_num_accepted_tokens_total") or 0
    if d:
        print(f"  MTP draft tokens        = {d:+.0f}")
        print(f"  MTP accepted tokens     = {a:+.0f}")
        print(f"  MTP acceptance (wave)   = {a/d*100:.1f}%")
    print("\nthankyou, come again")


if __name__ == "__main__":
    main()