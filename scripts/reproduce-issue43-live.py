#!/usr/bin/env python3
"""Reproduce issue #43 on the LIVE dspark server (issue #27 fix already in).

Six cold cells from the issue: 32K/62K x2/x4/x8. Exact protocol:
  - unique cold prefixes per lane (cache-hit delta 0)
  - max=min=256, ignore_eos=true, temperature=0, seed per lane, streaming+usage
  - decode rate measured TWO ways (issue #43 ask #4 distinction):
      * whole-request  = output_tokens / (last_token - first_request_send)
                         (the reporter's metric; includes TTFT stagger)
      * post-TTFT      = output_tokens / (last_token - first_token)
                         (decode-window-only; excludes prefill wait)
  - per-lane p95 ITL from inter-arrival times of streamed tokens
  - min/max ratio for both metrics across lanes
Reports counter deltas (preemptions, prompt/generation tokens, prefix hits,
MTP) per wave. Pass --dump-diag to also grep the container log for
[issue43-step ...] scheduler-diag lines emitted when DSPARK_ISSUE43_SCHED_DIAG=1.

Usage:
  python3 scripts/reproduce-issue43-live.py            # all six cells
  python3 scripts/reproduce-issue43-live.py 32768 8    # one cell: 32K x8
"""
import json, time, urllib.request, sys, threading, statistics, argparse, subprocess, os

API = os.environ.get("DSPARK_API", "http://127.0.0.1:8888")
MODEL = os.environ.get("DSPARK_MODEL", "deepseek-v4-flash-0731")
OUT = 256
SEED_BASE = 43_000
CONTAINER = os.environ.get("DSPARK_CONTAINER", "deepseek-v4-flash-vllm-dspark-1")

CELLS = [(32768, 2), (32768, 4), (32768, 8),
         (63488, 2), (63488, 4), (63488, 8)]


def post(url, body, timeout=900):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    return urllib.request.urlopen(req, timeout=timeout)


def tokenize(text):
    with post(f"{API}/tokenize", {"model": MODEL, "prompt": text}, timeout=120) as r:
        return json.load(r)["count"]


def build_unique_prompt(target, nonce):
    front = (f"issue-43 lane-nonce-{nonce} " +
             "ABCDEFGHJKLMNPQRSTUVWXYZ0123456789" * 64)
    unit = "benchmark context datum alpha beta gamma delta epsilon "
    # Tokenize front + one unit to size the unit, then size reps to land
    # near the target (the old `target // 3` heuristic overshot to ~2.7x).
    front_tok = tokenize(front + unit)
    unit_tok = tokenize(front + unit + unit) - front_tok
    reps = max(1, (target - front_tok) // max(1, unit_tok))
    text = front + "\n\n" + unit * reps
    # trim loop: never overshoot by more than one unit; add one unit if under.
    while True:
        c = tokenize(text)
        if c >= target and c <= target + unit_tok:
            return text, c
        if c > target + unit_tok:
            reps = max(1, reps - 1)
            text = front + "\n\n" + unit * reps
            continue
        reps += 1
        text = front + "\n\n" + unit * reps


def run_lane(idx, prompt, prompt_len, out):
    res = {"idx": idx, "prompt_tokens": prompt_len,
           "ttft": None, "first_byte": None, "done": None,
           "itls": [], "out_tokens": 0, "error": None, "usage": None}
    body = {"model": MODEL, "prompt": prompt,
            "max_tokens": OUT, "min_tokens": OUT,
            "ignore_eos": True, "temperature": 0.0,
            "seed": SEED_BASE + idx, "stream": True,
            "stream_options": {"include_usage": True}}
    t_send = time.perf_counter()
    last_tok_t = None
    try:
        with post(f"{API}/v1/completions", body) as resp:
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
                        now = time.perf_counter()
                        if res["first_byte"] is None:
                            res["first_byte"] = now
                            res["ttft"] = now - t_send
                        else:
                            res["itls"].append((now - last_tok_t) * 1000.0)
                        last_tok_t = now
                        res["out_tokens"] += 1
                    if ev.get("usage"):
                        res["usage"] = ev["usage"]
    except Exception as e:
        res["error"] = repr(e)
    res["done"] = time.perf_counter()
    res["wall"] = res["done"] - t_send
    if res["first_byte"]:
        # authoritative OUT-token decode window (server emits exactly OUT).
        res["decode_secs"] = res["done"] - res["first_byte"]
        res["post_ttft_tps"] = OUT / res["decode_secs"] if res["decode_secs"] > 0 else 0.0
        res["whole_tps"] = OUT / res["wall"] if res["wall"] > 0 else 0.0
        res["p95_itl_ms"] = (statistics.quantiles(res["itls"], n=20)[18]
                             if len(res["itls"]) >= 20
                             else (statistics.mean(res["itls"]) if res["itls"] else 0.0))
    return res


def metric(m, txt):
    for line in txt.splitlines():
        if line.startswith(m + "{") and "_created" not in line:
            i = line.rfind("}")
            if i >= 0:
                return float(line[i + 2:])
    return None


def run_cell(n_lanes, target_in, diag_log=None):
    print(f"\n=== cell {target_in//1024}K x{n_lanes} (cold) ===")
    m_before = urllib.request.urlopen(API + "/metrics", timeout=30).read().decode()
    prompts = [build_unique_prompt(target_in, f"l{i}-{int(time.time())}")
               for i in range(n_lanes)]
    for i, (_, c) in enumerate(prompts):
        print(f"  lane{i} prompt tokens = {c}")
    wave_s = time.perf_counter()
    results = [None] * n_lanes

    def worker(i):
        results[i] = run_lane(i, prompts[i][0], prompts[i][1], OUT)

    ts = [threading.Thread(target=worker, args=(i,)) for i in range(n_lanes)]
    for t in ts: t.start()
    for t in ts: t.join()
    print(f"  wave wall = {time.perf_counter()-wave_s:.3f} s")
    print(f"{'lane':>4} {'in':>7} {'TTFT(s)':>8} {'whole_t/s':>9} "
          f"{'postTTFT_t/s':>13} {'p95ITL(ms)':>10} {'out':>5}")
    whole, post = [], []
    for r in results:
        if not r or r.get("error"):
            print(f"  lane{r['idx'] if r else '?'} ERROR {r.get('error') if r else 'no result'}")
            continue
        whole.append(r.get("whole_tps", 0))
        post.append(r.get("post_ttft_tps", 0))
        print(f"{r['idx']:>4} {r['prompt_tokens']:>7} "
              f"{(r['ttft'] or 0):>8.3f} {r.get('whole_tps',0):>9.2f} "
              f"{r.get('post_ttft_tps',0):>13.2f} {r.get('p95_itl_ms',0):>10.1f} "
              f"{r['out_tokens']:>5}")
    if len(whole) >= 2:
        wr = min(whole) / max(whole) if max(whole) > 0 else 0
        pr = min(post) / max(post) if max(post) > 0 else 0
        print(f"  whole  min/max = {wr:.3f}   postTTFT min/max = {pr:.3f}")
    time.sleep(2)
    m_after = urllib.request.urlopen(API + "/metrics", timeout=30).read().decode()

    def delta(nm):
        return (metric(nm, m_after) or 0) - (metric(nm, m_before) or 0)
    print(f"  [deltas] preempts={delta('vllm:num_preemptions_total'):+.0f} "
          f"prompt={delta('vllm:prompt_tokens_total'):+.0f} "
          f"gen={delta('vllm:generation_tokens_total'):+.0f} "
          f"hits={delta('vllm:prefix_cache_hits_total'):+.0f}")
    d = delta("vllm:spec_decode_num_draft_tokens_total") or 0
    a = delta("vllm:spec_decode_num_accepted_tokens_total") or 0
    if d:
        print(f"  [mtp] draft={d:+.0f} accepted={a:+.0f} rate={a/d*100:.1f}%")
    return {"whole": whole, "post": post}


def dump_diag():
    """Grep container log for [issue43-step ...] lines (last 2000)."""
    try:
        out = subprocess.run(
            ["docker", "logs", "--tail", "2000", CONTAINER],
            capture_output=True, text=True, timeout=20).stdout
    except Exception as e:
        print(f"  [diag] could not read container logs: {e}")
        return
    lines = [l for l in out.splitlines() if "[issue43-step" in l]
    if not lines:
        print("  [diag] no [issue43-step] lines found "
              "(set DSPARK_ISSUE43_SCHED_DIAG=1 and restart).")
        return
    print(f"  [diag] {len(lines)} step lines emitted. Sample (first 5):")
    for l in lines[:5]:
        print("   ", l.strip())
    skips = sum("decode_skips=[]" not in l for l in lines)
    print(f"  [diag] steps with non-empty decode_skips = "
          f"{sum('decode_skips=[]' not in l for l in lines)}/{len(lines)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cells", nargs="*", type=int,
                    help="optional <target_in> <n_lanes> for a single cell")
    ap.add_argument("--dump-diag", action="store_true",
                    help="grep container logs for [issue43-step] diag lines")
    args = ap.parse_args()
    cells = [tuple(args.cells)] if len(args.cells) == 2 else CELLS
    summary = []
    for tin, n in cells:
        r = run_cell(n, tin)
        if r["whole"] and r["post"]:
            summary.append((tin, n,
                            min(r["whole"]) / max(r["whole"]) if max(r["whole"]) else 0,
                            min(r["post"]) / max(r["post"]) if max(r["post"]) else 0))
    print("\n=== summary ===")
    print(f"{'shape':>10} {'whole min/max':>14} {'postTTFT min/max':>18}")
    for tin, n, wr, pr in summary:
        print(f"{tin//1024}K x{n:<2} {wr:>14.3f} {pr:>18.3f}")
    if args.dump_diag:
        dump_diag()
    print("\nthankyou, come again")


if __name__ == "__main__":
    main()