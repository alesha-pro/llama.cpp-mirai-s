# Mirai S in llama.cpp

This fork runs Mirai S Qwen3.8-27B (`trymirai/Qwen3.8-27B-S-experimental`, 2.4-bit QTIP-style trellis) from a GGUF
that keeps Mirai's compressed codes bit for bit. The weights are not re-quantized to a llama.cpp type. The fork adds
the model's codec to ggml, and the CUDA kernels are ports of Mirai's vLLM plugin (`mirai_s` 0.2.1, Apache-2.0).

Base: llama.cpp `d834d44e6`.

## Quick start

A ready GGUF (and a vision mmproj) is on Hugging Face:
[anonymousmaharaj/Qwen3.8-27B-S-mirai-GGUF](https://huggingface.co/anonymousmaharaj/Qwen3.8-27B-S-mirai-GGUF).

```bash
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DCUDAToolkit_ROOT=/usr/local/cuda
cmake --build build -j --target llama-server
hf download anonymousmaharaj/Qwen3.8-27B-S-mirai-GGUF --local-dir qwen3.8-s

# 12 GB card, 128K context (q4_0 KV), 11.3 GB peak
./build/bin/llama-server -m qwen3.8-s/Qwen3.8-27B-S-mirai.gguf -ngl 99 -fa on --jinja \
  -c 131072 -ctk q4_0 -ctv q4_0 -b 1024 -ub 1024
# 12 GB card, 74K context with q8_0 KV, 11.1 GB peak
./build/bin/llama-server -m qwen3.8-s/Qwen3.8-27B-S-mirai.gguf -ngl 99 -fa on --jinja \
  -c 73728 -ctk q8_0 -ctv q8_0 -b 1024 -ub 1024
# 16 GB card, 128K context with MTP speculative decoding, 14.6 GB peak
./build/bin/llama-server -m qwen3.8-s/Qwen3.8-27B-S-mirai.gguf -ngl 99 -fa on --jinja \
  -c 131072 -ctk q8_0 -ctv q8_0 --spec-type draft-mtp --spec-draft-n-max 3
# vision with the encoder on the CPU (0 VRAM): add to any line above
  --mmproj qwen3.8-s/mmproj-Qwen3.8-27B-base-f16.gguf --no-mmproj-offload -t <physical cores>
```

Mirai's checkpoint ships the language model only. The mmproj is the vision encoder of the base Qwen3.8-27B, converted
with the stock `convert_hf_to_gguf.py --mmproj`; the compressed language model reads its embeddings fine (charts, UI
text, scene descriptions in my checks).

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

The 128K q4_0 setup decodes 30.4 tok/s at 117K. On the same card at 300 W, Mirai's vLLM plugin (0.2.1) is faster on
short prompts: 44 vs 39 tok/s decode, 1336 vs 1120 prefill, 107 vs 84 on code with MTP 3. In a 12 GB budget it fits
37.6K tokens of context with bf16 KV and 74.4K with fp8 KV.

The long-context changes, each against its off switch on the same binary (llama-bench, q8_0 KV): decode at 64K
27.9 → 35.2 tok/s (q4_0 KV 26.6 → 34.8), prefill of 2048 tokens 1103 → 1225 tok/s.
