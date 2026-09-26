#!/usr/bin/env python3
"""Greedy continuations of the same token ids from Mirai's vLLM plugin and the mirai-s llama.cpp fork (or a second
vLLM server, --vllm-b).

Both run the same compressed weights, so the continuations should agree token for token until a near-tie; the report
gives the length of the common prefix and, at the first position, the top-5 overlap and the logprob gap.

  python3 compare_vllm.py --vllm http://localhost:18293 --llama http://localhost:18400
"""
import argparse
import json
import math
import urllib.request

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    "Photosynthesis is the process by which",
    "Q: What is 17 * 23?\nA:",
    "Once upon a time, in a small village by the sea,",
    "В 1961 году Юрий Гагарин",
    "The three laws of thermodynamics are",
    "SELECT name, COUNT(*) FROM orders",
]
# long prompts (~7k tokens, the speedcheck log): compressed KV caches differ from bf16 most with a long context
LOG = "\n".join(f"Entry {i}: station {i % 23} logged {i * 37 % 101} crates of item {i * 13 % 29}." for i in range(330))
LONG_PROMPTS = [
    LOG + "\n\nQuestion: how many crates did entry 17 log, and at which station? Answer:",
    LOG + "\n\nWhich item numbers appear most often in the log above? Explain briefly. Answer:",
]


def post(url: str, body: dict) -> dict:
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vllm", default="http://localhost:18293")
    ap.add_argument("--llama", default="http://localhost:18400")
    ap.add_argument("--vllm-b", help="compare against a second vLLM server instead of llama-server")
    ap.add_argument("--model", default="qwen3.8-s")
    ap.add_argument("--tokens", type=int, default=48)
    ap.add_argument("--long", action="store_true", help="add the two ~7k-token prompts")
    args = ap.parse_args()

    summary = []
    for prompt in PROMPTS + (LONG_PROMPTS if args.long else []):
        ids = post(f"{args.vllm}/tokenize", {"model": args.model, "prompt": prompt})["tokens"]
        v = post(f"{args.vllm}/v1/completions", {
            "model": args.model, "prompt": ids, "max_tokens": args.tokens, "temperature": 0, "logprobs": 5,
            "return_tokens_as_token_ids": True})["choices"][0]
        v_ids = [int(t.split(":")[1]) for t in v["logprobs"]["tokens"]]
        v_top = {int(k.split(":")[1]): lp for k, lp in v["logprobs"]["top_logprobs"][0].items()}

        if args.vllm_b:
            b = post(f"{args.vllm_b}/v1/completions", {
                "model": args.model, "prompt": ids, "max_tokens": args.tokens, "temperature": 0, "logprobs": 5,
                "return_tokens_as_token_ids": True})["choices"][0]
            l_ids = [int(t.split(":")[1]) for t in b["logprobs"]["tokens"]]
            l_top = {int(k.split(":")[1]): lp for k, lp in b["logprobs"]["top_logprobs"][0].items()}
        else:
            l = post(f"{args.llama}/completion", {
                "prompt": ids, "n_predict": args.tokens, "temperature": 0, "top_k": 1, "n_probs": 5,
                "cache_prompt": False, "return_tokens": True, "samplers": ["top_k"]})
            l_ids = l["tokens"]
            l_top = {t["id"]: t["logprob"] for t in l["completion_probabilities"][0]["top_logprobs"]}

        common = 0
        while common < min(len(v_ids), len(l_ids)) and v_ids[common] == l_ids[common]:
            common += 1
        shared = set(v_top) & set(l_top)
        gap = max((abs(v_top[t] - l_top[t]) for t in shared), default=float("nan"))
        p_v = {t: math.exp(lp) for t, lp in v_top.items()}
        rec = {"prompt": prompt[:40] if len(prompt) < 200 else "LONG ..." + prompt[-40:], "prompt_tokens": len(ids), "common_prefix": common, "of": min(len(v_ids), len(l_ids)),
               "top5_overlap": len(shared), "max_logprob_gap_top5": round(gap, 4),
               "vllm_top1_p": round(max(p_v.values()), 4)}
        summary.append(rec)
        print(json.dumps(rec, ensure_ascii=False), flush=True)
    full = sum(r["common_prefix"] == r["of"] for r in summary)
    print(json.dumps({"prompts": len(summary), "identical_continuations": full,
                      "mean_common_prefix": sum(r["common_prefix"] for r in summary) / len(summary)}))


if __name__ == "__main__":
    main()
