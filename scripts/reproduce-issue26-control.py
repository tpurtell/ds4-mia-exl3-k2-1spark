#!/usr/bin/env python3
"""Issue #26 single-lane control: one unique large prefix, cold then N warm
replays, must retain cache every warm round (reporter saw 20/20 at 256K/512K).
Confirms the #26 collapse is concurrency-specific and that our #27 fix did
not regress the single-lane retention path.
"""
import json, time, urllib.request, sys, threading

API = "http://127.0.0.1:8888"
MODEL = "deepseek-v4-flash-0731"
TARGET_IN = int(sys.argv[1]) if len(sys.argv) > 1 else 262144
WARM = int(sys.argv[2]) if len(sys.argv) > 2 else 5
OUT = 256


def post(url, body, timeout=900):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    return urllib.request.urlopen(req, timeout=timeout)


def tokenize(text):
    with post(f"{API}/tokenize", {"model": MODEL, "prompt": text}) as r:
        return json.load(r)["count"]


def build(target, nonce):
    front = f"issue-26 control nonce-{nonce} " + "ZYXWVUTSRQPONMLKJI" * 64
    unit = "control context datum alpha beta gamma delta epsilon zeta "
    # iterate tightly; grow in small increments to avoid 3x+ overshoot
    text = front + "\n\n"
    c = tokenize(text)
    while c < target:
        # add roughly the deficit in tokens; one `unit` repeat ≈ 11 tokens
        add = max(1, (target - c) // 11)
        text += unit * add
        c = tokenize(text)
    return text, c


def metric(name, m):
    for line in m.splitlines():
        if line.startswith(name + "{") and "_created" not in line:
            i = line.rfind("}")
            if i >= 0:
                return float(line[i + 2:])
    return None


def run(prompt, seed):
    body = {"model": MODEL, "prompt": prompt, "max_tokens": OUT,
            "min_tokens": OUT, "ignore_eos": True, "temperature": 0.0,
            "seed": seed, "stream": True,
            "stream_options": {"include_usage": True}}
    cached = 0
    with post(f"{API}/v1/completions", body) as resp:
        buf = b""
        while True:
            chunk = resp.read(4096)
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if line.startswith(b"data:"):
                    data = line[5:].strip()
                    if data == b"[DONE]":
                        continue
                    try:
                        ev = json.loads(data)
                    except Exception:
                        continue
                    if ev.get("usage"):
                        pd = ev["usage"].get("prompt_tokens_details") or {}
                        cached = pd.get("cached_tokens", 0)
    return cached


def main():
    p, c = build(TARGET_IN, f"ctrl-{int(time.time())}")
    print(f"[control] target={TARGET_IN} actual_in={c} warm_rounds={WARM}")
    mb = urllib.request.urlopen(API + "/metrics").read().decode()
    t0 = time.perf_counter()
    cold_c = run(p, 2600)
    print(f"  cold cached_tokens = {cold_c} (expect 0)  wall={time.perf_counter()-t0:.2f}s")
    hits = 0
    for i in range(WARM):
        mb2 = urllib.request.urlopen(API + "/metrics").read().decode()
        t0 = time.perf_counter()
        cc = run(p, 2600)
        wa = urllib.request.urlopen(API + "/metrics").read().decode()
        d = (metric("vllm:prefix_cache_hits_total", wa) or 0) - \
            (metric("vllm:prefix_cache_hits_total", mb2) or 0)
        ok = (cc > 0 and d > 0)
        hits += int(ok)
        print(f"  warm{i+1}: cached_tokens={cc} server_hits_delta={d:+.0f} "
              f"{'HIT' if ok else 'MISS'}  wall={time.perf_counter()-t0:.2f}s")
    print(f"\n  warm hits: {hits}/{WARM}")


if __name__ == "__main__":
    main()