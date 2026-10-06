# Mirai S in llama.cpp

This fork runs Mirai S Qwen3.8-27B (`trymirai/Qwen3.8-27B-S-experimental`, 2.4-bit QTIP-style trellis) from a GGUF
that keeps Mirai's compressed codes bit for bit. The weights are not re-quantized to a llama.cpp type. The fork adds
the model's codec to ggml, and the CUDA kernels are ports of Mirai's vLLM plugin (`mirai_s` 0.2.1, Apache-2.0).

Base: llama.cpp `d834d44e6`.

## Quick start

A ready GGUF (and a vision mmproj) is on Hugging Face:
[alesha-pro/Qwen3.8-27B-S-mirai-GGUF](https://huggingface.co/alesha-pro/Qwen3.8-27B-S-mirai-GGUF).

```bash
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DCUDAToolkit_ROOT=/usr/local/cuda
cmake --build build -j --target llama-server
hf download alesha-pro/Qwen3.8-27B-S-mirai-GGUF --local-dir qwen3.8-s

# 12 GB card, 128K context (q4_0 KV), 11.3 GB peak
./build/bin/llama-server -m qwen3.8-s/Qwen3.8-27B-S-mirai.gguf -ngl 99 -fa on -np 1 --jinja \
  -c 131072 -ctk q4_0 -ctv q4_0 -b 1024 -ub 1024
# 12 GB card, 74K context with q8_0 KV, 11.1 GB peak
./build/bin/llama-server -m qwen3.8-s/Qwen3.8-27B-S-mirai.gguf -ngl 99 -fa on -np 1 --jinja \
  -c 73728 -ctk q8_0 -ctv q8_0 -b 1024 -ub 1024
# 16 GB card, 128K context with MTP speculative decoding, 14.6 GB peak
./build/bin/llama-server -m qwen3.8-s/Qwen3.8-27B-S-mirai.gguf -ngl 99 -fa on -np 1 --jinja \
  -c 131072 -ctk q8_0 -ctv q8_0 --spec-type draft-mtp --spec-draft-n-max 3
# vision with the encoder on the CPU (0 VRAM): add to any line above
  --mmproj qwen3.8-s/mmproj-Qwen3.8-27B-base-f16.gguf --no-mmproj-offload -t <physical cores>
```

If an older CUDA is first on your PATH, add `-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc` to the first line.

`-np 1` matters on this hybrid model: every server slot keeps its own DeltaNet state, and the default slot count
adds about 450 MB (11.7 GB instead of 11.3 GB at 128K), which leaves a 12 GB card almost nothing.

Mirai's checkpoint ships the language model only. The mmproj is the vision encoder of the base Qwen3.8-27B, converted
with the stock `convert_hf_to_gguf.py --mmproj`; the compressed language model reads its embeddings fine (charts, UI
text, scene descriptions in my checks).

## The full 262K window on 12 GB: mirai-s-ada

This fork keeps the KV cache in VRAM, so a 12 GB card stops at 128K with q4_0 KV (147K with `-b 512 -ub 512`).
[mirai-s-ada](https://github.com/professorpalmer/mirai-s-ada) by Cary Palmer ports the codec to a serving engine with a
tiered cache: about 44K positions in VRAM and the rest in pinned system RAM. That gives the full 262,144 window at
q8_0, with MTP drafting at every depth. His numbers on an RTX 4070 12 GB: 75.8 tok/s on a fresh chat, 40.3 at 60K. I
checked his engine on Ubuntu with a 3090 held to the same 12 GB recipe: its greedy output matches this fork on 5 of 5
prompts, 66.9 tok/s at 8K and 27.9 at 60K. If you want the long window or the speed, use that project. This fork is
the smaller one to read, to rebase, and to fall back to.

## Removing refusals

The weights are trellis codes, so the usual abliteration (editing the matrices) cannot be written back. Instead the
refusal direction is a control vector file, and the fork subtracts it from the residual stream after every layer
while the model runs: `h -= (h.v) v`. Add 2 arguments to any `llama-server` line above:

```bash
--control-vector-scaled qwen3.8-s/Qwen3.8-27B-S-mirai-refusal-direction.gguf:1.0 --cvec-mode project
```

`--cvec-mode add` is upstream's behaviour and the default. Without the flag the output is unchanged: KL to the build
before this change is 0.000000 and the top token is the same in 100% of positions.

On this fork (GGUF template, no system message, one RTX 3090), a refusal counted by a regex on the start of the answer:

| | without the vector | with the vector |
|---|---:|---:|
| 64 harmful instructions (AdvBench, not used for the direction), thinking off | 63 refused | 0 |
| 82 held-out behaviours (JailbreakBench, non-AdvBench rows), thinking off | not run | 0 |
| 24 harmful instructions, thinking on | 23 | 0 (2 answers empty at the token limit) |
| 32 ordinary instructions refused | 0 | 0 |
| decode on short prompts, tok/s | 39.9 | 39.2 |

KL to the model without the vector: 0.019 on prose, 0.009 on code. The top token stays the same in 94.0% and 97.1% of
positions. The same file behind mirai-s-ada's template, which always writes a system message: 0 of 64, 0 of 82, and 1
of 24 with thinking on.

The catch: the counts come from a regex over 24 to 82 prompts, I did not read the answers one by one. A very
different system prompt may leave more refusals.

How the file is made, so you can repeat it or make one for another model:

```bash
# 1. the residual stream after every layer at the last prompt token; prompts.txt = one templated prompt per line
./build/bin/llama-resid-dump -m model.gguf --positive-file harmful.txt  -o harmful.bin  -ngl 99 -c 2048
./build/bin/llama-resid-dump -m model.gguf --positive-file harmless.txt -o harmless.bin -ngl 99 -c 2048
# 2. one --pair per prompt context (thinking off, thinking on, with a system message ...)
python3 tools/mirai-s/refusal_direction.py vector.gguf --pair harmful.bin harmless.bin
```

Per layer and context the script takes the difference of means, removes its component along the mean harmless state,
normalizes, and averages the contexts. The published file used 256 AdvBench and 256 Alpaca instructions in 6 contexts
(this template and mirai-s-ada's, thinking on and off, 2 system texts) and has a direction for each of layers 1 to 63.
With those dumps the script reproduces it byte for byte. Two things I ran into: the plain difference of means broke
the model in projection mode (empty answers), because an ordinary prompt has 15% to 54% of its norm along it. And the
directions for thinking on and thinking off differ, so a vector from one prompt ending left 54 of 64 refusals at the
other.

## Files

| Path | What |
|---|---|
| `tools/mirai-s/convert_mirai_s_to_gguf.py` | the `vllm/` folder of the HF repo (sidecar `trellis.mirai` + dense shards) to GGUF |
| `tools/mirai-s/mirai_s_decode.py` | NumPy reference decoder of the trellis tapes and the rotation |
| `tools/mirai-s/check_decoder.py` | cross-check of that decoder against lalamo's own decode of the same rows |
| `ggml/src/ggml-cpu/mirai-s.cpp` | CPU implementation, the reference for the CUDA kernels (layouts and math documented there) |
| `ggml/src/ggml-cuda/mirai-s.cu` | CUDA kernels |
| `tests/test-backend-ops.cpp` | `test_mirai_s`: CUDA against CPU on random codes, every path and format |
| `tools/mirai-s/compare_vllm.py` | greedy continuations and top-5 logprobs against Mirai's vLLM plugin |
| `tools/resid-dump/` | `llama-resid-dump`: the residual stream after every layer at the last token of each prompt |
| `tools/mirai-s/refusal_direction.py` | residual dumps to a refusal-direction control vector |
| `src/llama-adapter.cpp` | the control vector in projection mode (`--cvec-mode project`) |

## Format

Four ggml types, ids 90-93 (kept away from upstream's range so a GGUF survives rebases):

| Type | Block | Bytes | Use |
|---|---:|---:|---|
| `MS_V4T8` | 64 columns | 16 + 1 | most FFN and output projections, ~2 bits/weight |
| `MS_V2T4` | 64 | 16 + 2 | layers 0 and a few others, ~2 bits/weight |
| `MS_V2T6` | 128 | 48 + 2 | attention and DeltaNet input projections, ~3 bits/weight |
| `MS_I3` | 128 | 48 + 1 | the output head, 3-bit codes with a 16-step ladder per 64 columns |

Rows are grouped by 32 and interleaved (lane = row), so a warp's packet load is one 512-byte read. Every trellis
weight has `<name>.scale` (F32, one value per output row). Model-wide tensors: `mirai.rot.{5120,6144,17408}` (input
signs + small_q of the rotation), `mirai.head_aux` (head input signs + ladder), KV `mirai.codebook.v4/v2`.

Two layout differences from stock Qwen3.5 GGUFs:

- Full-attention `q_proj` is split into `attn_q` (queries) and `attn_gate` (output gate): Mirai stores the gate rows in
  their own trellis block, sometimes in another format.
- DeltaNet `ssm_out` keeps the checkpoint's grouped V-head column order; the graph permutes its input instead.

`ssm_alpha/beta` (48 rows) are decoded to F16, `token_embd` (Mirai's D4 embedding) to F16 on the CPU side, and the MTP
block (bf16 in the checkpoint) is written as Q8_0.

## Kernels

Each trellis matmul is `ggml_mirai_quantize` (rotate the input, quantize it per token to two int8 planes, q0 + q1/254)
followed by `ggml_mirai_mul_mat`. Inputs shared by several weights are quantized once per graph.

| Tokens | Path |
|---|---|
| 1 | dp4a GEMV, trellis decoded in registers |
| 2-384 | int8 tensor-core MMA (m16n8k32) on decoded 64-column slabs, 8-64 tokens per CTA |
| > 384 | weights decoded to int8 levels in 32 MiB chunks, cuBLASLt int8 GEMM, output epilogue |
| head | fp16 tensor-core MMA against `H32(signs * x)` |

All three trellis paths compute the same exact int32 dot products, so a token's output does not depend on its batch.
Tunables for experiments: `GGML_MIRAI_MMA_TOKENS` (384), `GGML_MIRAI_SPLIT_TOKENS` (16), `GGML_MIRAI_LEVELS_MIB` (32).

## Long context and agent turns

Three changes outside the codec, each with a switch to get upstream behavior back:

| Change | Files | Off switch |
|---|---|---|
| FA vector kernel packs the Q heads of one K/V head (GQA) into a block | `ggml/src/ggml-cuda/fattn-vec.cuh` | `GGML_CUDA_FA_VEC_GQA=0` |
| Chunked gated delta rule for prefill | `ggml/src/ggml-cuda/gated_delta_net.cu` | `GGML_CUDA_GDN_CHUNKED=0` |
| Prompt cache matched by text where the tokens differ | `tools/server/server-common.cpp`, `server-context.cpp` | `LLAMA_SERVER_TEXT_ALIGN=0` |

- **Decode with q8_0/q4_0 KV.** On Ampere, single-token attention over a quantized cache goes to the vector kernel,
  one block per Q head, so with GQA 6:1 every K/V row was read six times. The kernel takes a second column dimension
  (`ncols2`, heads of the group) and reads each row once; each KQ dot uses 16 threads instead of 32, halving the
  reduction shuffles that bound it. Quantized K/V only (f16 keeps the MMA kernel). q8_0 decode at 64K context:
  18.0 → 24.8 tok/s, 32K: 23.3 → 27.8 (220 W). Remaining cost: ~400 µs per layer at 64K against ~170 µs of pure
  reads, the kernel is bound by L1/shuffle traffic at 17% occupancy (255 registers).
- **Prefill.** The token-by-token DeltaNet kernel took ~15% of prefill (8.6 ms per layer per 2048 tokens). The chunked
  kernel follows FLA's `chunk_gated_delta_rule` in fp32: chunks of 32 tokens, `(I + A) Δ = β(V − D K S0)` solved by
  forward substitution, one block per head and 32-column slice of the state. Used for ≥ 64 tokens; with rollback
  snapshots (MTP) the last K tokens go through the old kernel, which writes the snapshots. Prefill of 6144 tokens:
  694 → 816 tok/s (220 W).
- **Prompt cache.** With sampling, the model now and then emits a token split that re-tokenizing the same text does not
  give (about one in a thousand tokens: `" Birds"+"ong"` for `" Bird"+"song"`, emoji bytes). The next request's
  re-rendered history then diverges from the cache there, and a hybrid model can only roll back to a checkpoint, so
  after a long answer the next agent turn re-prefilled everything since the previous prompt (30-54K tokens). The server now walks such spans piece by piece and, where the
  text is equal, takes the cached tokens for the prompt (control and user-defined tokens must match exactly).

## Build and convert

```bash
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DCUDAToolkit_ROOT=/usr/local/cuda -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
cmake --build build -j --target llama-server llama-bench test-backend-ops

python3 tools/mirai-s/convert_mirai_s_to_gguf.py <hf-repo>/vllm --outfile Qwen3.8-27B-S-mirai.gguf \
  --verify-base <bf16 Qwen3.8-27B>   # optional: correlates decoded rows with the base model

```

Link against CUDA 12 cuBLASLt: with cuBLAS 11 the int8 GEMM falls back to a tile that is about half as fast.

## Verification

- Decoder: `layers.10.mlp.down_proj` from the sidecar equals lalamo's decode of the same rows to 3e-8 (relative L2).
- Converter: all 416 trellis matrices correlate with the bf16 base model's rows at 0.92-0.99 (by format: V2T6 0.985,
  V4T8 0.953, V2T4 0.935 mean), so every row lands in the right place.
- Kernels: `test-backend-ops -o 'MIRAI.*'`, 33/33 (gemv, mma n8-n64, cuBLASLt path, all formats, head).
- End to end: greedy continuations of 8 prompts (English, code, Russian, SQL, arithmetic), 48 tokens each, are
  identical to Mirai's vLLM plugin, with and without MTP; top-5 logprobs overlap fully, max gap 0.056
  (`tools/mirai-s/compare_vllm.py`).
- Long-context changes: `test-backend-ops -o FLASH_ATTN_EXT` passes with new single-token cases for GQA 2-12 and
  q8_0/q4_0/f16 K/V; `-o GATED_DELTA_NET` 46/46 with chunked cases (chunk tails, head 128 with q/k broadcast, 2048
  tokens, rollback snapshots). Greedy vs vLLM with 2 long prompts: 10/10 identical with the changes off, 8/10 with GQA
  packing (the Russian prompt splits at a 0.0007 logprob tie), 9/10 with chunked prefill.

## Speed and VRAM (one RTX 3090 at 300 W)

llama-server, Mirai's `speedcheck.py` (6.9K-token prompt) and a long-prompt script, peak VRAM from `nvidia-smi`:

| Setup | Peak VRAM | Decode, fresh chat | Decode at 62K | Prefill |
|---|---:|---:|---:|---:|
| 128K, q4_0 KV, `-ub 1024` | 11.3 GB | 39.7 tok/s | 34.6 tok/s | 1008 tok/s |
| 74K, q8_0 KV, `-ub 1024` | 11.1 GB | 40.0 tok/s | 34.9 tok/s | 1009 tok/s |
| 128K, q8_0 KV, MTP 3 | 14.6 GB | 85 code / 57 prose | | 806 tok/s |

With `-b 512 -ub 512`, a refusal vector loaded and the image encoder on the CPU, the q4_0 setup fits 147,456 tokens (11,689 MiB
peak, 29.1 tok/s at 144K) and the q8_0 setup 81,920 (11,497 MiB, 32.9 tok/s at 78K); 163,840 at q4_0 peaks at 12,059 MiB.

The 128K q4_0 setup decodes 30.4 tok/s at 117K. On the same card at 300 W, Mirai's vLLM plugin (0.2.1) is faster on
short prompts: 44 vs 39 tok/s decode, 1336 vs 1120 prefill, 107 vs 84 on code with MTP 3. In a 12 GB budget it fits
37.6K tokens of context with bf16 KV and 74.4K with fp8 KV.

The long-context changes, each against its off switch on the same binary (llama-bench, q8_0 KV): decode at 64K
27.9 → 35.2 tok/s (q4_0 KV 26.6 → 34.8), prefill of 2048 tokens 1103 → 1225 tok/s.
