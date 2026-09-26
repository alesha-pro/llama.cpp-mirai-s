#!/usr/bin/env python3
"""Convert Mirai S Qwen3.8-27B (the `vllm/` folder of trymirai/Qwen3.8-27B-S-experimental) to GGUF for the mirai-s
llama.cpp fork.

The trellis matrices keep Mirai's compressed codes bit for bit. Their 32-row interleaved packets (the layout of Mirai's
vLLM plugin) are regrouped into llama.cpp's tensors: the row order of every tensor is learned by running the stock
Qwen3.5 converter on row-index probe tensors, so any row permutation it applies (V-head reordering) is reproduced
exactly. The stock converter also handles every dense tensor (norms, conv, A_log, dt_bias, the MTP block).

Written per model:
  blk.N.<proj>.weight   MS_V4T8 / MS_V2T4 / MS_V2T6   compressed rows, opaque 32-row interleaved layout
  blk.N.<proj>.scale    F32 [rows]                   Mirai's per-row scale
  blk.N.attn_gate       (full attention)             the output-gate rows of q_proj, split off attn_q because Mirai
                                                     stores them in their own trellis block
  blk.N.ssm_alpha/beta  F16                          48-row gates, decoded (rows are not a multiple of 32)
  output.weight         MS_I3                        the 3-bit head; output.scale F32 [vocab]
  token_embd.weight     F16                          the D4 embedding, decoded (lookups run on the CPU)
  mirai.rot.<K>         F32 [K + order^2]            input rotation: signs, then small_q row-major
  mirai.head_aux        F32 [n_embd + 16]            head input signs, then the 16-entry ladder
  KV mirai.codebook.v4/v2 [c, d0, d1, d2, d3]
The trunk's ssm_out columns stay in checkpoint (grouped V-head) order; the graph permutes its input instead.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(1, str(ROOT / "gguf-py"))
sys.path.insert(1, str(ROOT))
sys.path.insert(1, str(Path(__file__).resolve().parent))
import gguf  # noqa: E402
from safetensors import safe_open  # noqa: E402

from conversion.qwen import Qwen3_5TextModel  # noqa: E402
import mirai_s_decode as md  # noqa: E402

logger = logging.getLogger("mirai-s")

QTYPE = {"v4t8": gguf.GGMLQuantizationType.MS_V4T8, "v2t4": gguf.GGMLQuantizationType.MS_V2T4,
         "v2t6": gguf.GGMLQuantizationType.MS_V2T6}
PREFIX = "model.layers."  # after TextModel.filter_tensors drops "language_model."


class Sidecar:
    def __init__(self, path: Path) -> None:
        self.f = safe_open(str(path), "np")
        self.layers = json.loads(self.f.metadata()["layers"])
        self._blocks: dict[tuple[str, int], tuple] = {}

    def get(self, name: str) -> np.ndarray:
        return self.f.get_tensor(name)

    def _block(self, key: str, index: int):
        """Block `index` of layer `key` de-interleaved to per-row arrays: packets [rows, P, W, 16], entries [rows, P]."""
        cache_key = (key, index)
        if cache_key not in self._blocks:
            self._blocks.clear()  # one block at a time: they are large
            meta = self.layers[key]
            fmt = meta["blocks"][index]["format"]
            v, _, steps, words, edt = md.FORMATS[fmt]
            p = meta["in_features"] // (steps * v)
            rowscale = self.get(f"{key}.{index}.rowscale")
            g = rowscale.size // 32
            packets = self.get(f"{key}.{index}.packets").reshape(g, p, words, 32, 16).transpose(0, 3, 1, 2, 4)
            entries = self.get(f"{key}.{index}.entries").view(edt).reshape(g, p, 32).transpose(0, 2, 1)
            self._blocks[cache_key] = (fmt, packets.reshape(g * 32, p, words, 16), entries.reshape(g * 32, p), rowscale)
        return self._blocks[cache_key]

    def rows(self, key: str, out_cols: np.ndarray):
        """Rows for output columns `out_cols` of vLLM layer `key`: (format, packets, entries, rowscale)."""
        meta = self.layers[key]
        perm = self.get(f"{key}.perm").astype(np.int64) if meta["has_perm"] else np.arange(meta["out_features"])
        internal = perm[out_cols]
        fmts = set()
        result_p, result_e, result_s = None, None, np.empty(len(internal), np.float32)
        for index, block in enumerate(meta["blocks"]):
            rows_b = self.get(f"{key}.{index}.rowscale").size
            sel = np.nonzero((internal >= block["offset"]) & (internal < block["offset"] + rows_b))[0]
            if sel.size == 0:
                continue
            fmt, packets, entries, rowscale = self._block(key, index)
            fmts.add(fmt)
            local = internal[sel] - block["offset"]
            if result_p is not None and result_p.shape[1:] != packets.shape[1:]:
                raise ValueError(f"{key}: rows from mixed formats {fmts}")
            if result_p is None:
                result_p = np.empty((len(internal),) + packets.shape[1:], np.uint8)
                result_e = np.empty((len(internal), entries.shape[1]), entries.dtype)
            result_p[sel] = packets[local]
            result_e[sel] = entries[local]
            result_s[sel] = rowscale[local]
        assert len(fmts) == 1, f"{key}: rows from mixed formats {fmts}"
        return fmts.pop(), result_p, result_e, result_s


def interleave(packets: np.ndarray, entries: np.ndarray) -> np.ndarray:
    """Per-row packets [n, P, W, 16] + entries [n, P] -> the opaque tensor bytes, shaped [n, row_bytes] for gguf-py.
    Each group of 32 rows is self-contained (32 * row_bytes): its packets as [P][W][32 rows][16 B], so a warp with
    lane = row loads 512 contiguous bytes, then its entry states as [P][32 rows]."""
    n, p, w, _ = packets.shape
    assert n % 32 == 0, n
    g = n // 32
    pk = packets.reshape(g, 32, p, w, 16).transpose(0, 2, 3, 1, 4).reshape(g, -1)
    en = entries.astype(entries.dtype.newbyteorder("<")).reshape(g, 32, p).transpose(0, 2, 1).reshape(g, -1)
    return np.concatenate([pk.view(np.uint8), en.view(np.uint8)], axis=1).reshape(n, -1)


class MiraiSQwen35(Qwen3_5TextModel):
    """Stock Qwen3.5 conversion for dense tensors; trellis tensors are intercepted and written raw."""
    model_arch = gguf.MODEL_ARCH.QWEN35

    def __init__(self, *args, sidecar: Sidecar, verify_dir: Path | None, verify_rows: int, **kwargs) -> None:
        self.sidecar = sidecar
        self.verify_dir = verify_dir
        self.verify_rows = verify_rows
        self._probe_cache: dict[str, list] = {}
        self._raw: list[tuple[str, np.ndarray, gguf.GGMLQuantizationType]] = []
        self._verify_report: list[dict] = []
        super().__init__(*args, **kwargs)
        h = self.hparams
        n_k, n_v, dk, dv = h["linear_num_key_heads"], h["linear_num_value_heads"], h["linear_key_head_dim"], h["linear_value_head_dim"]
        qkv_lin = 2 * n_k * dk + n_v * dv
        q_rows = h["num_attention_heads"] * h["head_dim"] * 2
        kv_rows = h["num_key_value_heads"] * h["head_dim"]
        ff = h["intermediate_size"]
        # HF linear -> (vLLM fused layer, first output column, rows)
        self.hf2vllm = {
            "linear_attn.in_proj_qkv": ("linear_attn.in_proj_qkvz", 0, qkv_lin),
            "linear_attn.in_proj_z": ("linear_attn.in_proj_qkvz", qkv_lin, n_v * dv),
            "linear_attn.in_proj_b": ("linear_attn.in_proj_ba", 0, n_v),
            "linear_attn.in_proj_a": ("linear_attn.in_proj_ba", n_v, n_v),
            "linear_attn.out_proj": ("linear_attn.out_proj", 0, h["hidden_size"]),
            "self_attn.q_proj": ("self_attn.qkv_proj", 0, q_rows),
            "self_attn.k_proj": ("self_attn.qkv_proj", q_rows, kv_rows),
            "self_attn.v_proj": ("self_attn.qkv_proj", q_rows + kv_rows, kv_rows),
            "self_attn.o_proj": ("self_attn.o_proj", 0, h["hidden_size"]),
            "mlp.gate_proj": ("mlp.gate_up_proj", 0, ff),
            "mlp.up_proj": ("mlp.gate_up_proj", ff, ff),
            "mlp.down_proj": ("mlp.down_proj", 0, h["hidden_size"]),
        }
        # register the trellis-backed HF tensors so the stock pipeline hands them to modify_tensors
        self._trellis_names: set[str] = set()
        for key in self.sidecar.layers:
            m = re.match(r"layers\.(\d+)\.(.+)$", key)
            layer, role = int(m[1]), m[2]
            for hf, (vkey, _, _) in self.hf2vllm.items():
                if vkey == role:
                    name = f"{PREFIX}{layer}.{hf}.weight"
                    assert name not in self.model_tensors, name
                    self._trellis_names.add(name)
                    self.model_tensors[name] = lambda: torch.empty(0)

    def _probe(self, name: str, bid: int, rows: int, cols: int):
        """[(gguf name template, source HF row of each output row, columns permuted?)] for this kind of tensor."""
        kind = re.sub(r"layers\.\d+\.", "layers.N.", name)
        if kind not in self._probe_cache:
            r = torch.arange(rows, dtype=torch.float32)[:, None].expand(rows, cols).contiguous()
            c = torch.arange(cols, dtype=torch.float32)[None, :].expand(rows, cols).contiguous()
            out_r = list(super().modify_tensors(r, name, bid))
            out_c = list(super().modify_tensors(c, name, bid))
            result = []
            for (n1, t1), (n2, t2) in zip(out_r, out_c):
                assert n1 == n2, (n1, n2)
                t1, t2 = t1.reshape(t1.shape[0], -1), t2.reshape(t2.shape[0], -1)
                src_rows = t1[:, 0].long()
                assert torch.equal(t1, src_rows[:, None].float().expand_as(t1)), f"{name}: rows are not a pure gather"
                src_cols = t2[0, :].long()
                assert torch.equal(t2, src_cols[None, :].float().expand_as(t2)), f"{name}: cols are not a pure gather"
                col_perm = not torch.equal(src_cols, torch.arange(cols))
                result.append((n1.replace(f"blk.{bid}.", "blk.{bid}."), src_rows.numpy(), col_perm))
            self._probe_cache[kind] = result
        return [(tmpl.format(bid=bid), rows_, cp) for tmpl, rows_, cp in self._probe_cache[kind]]

    def modify_tensors(self, data_torch, name, bid):
        if name not in self._trellis_names:
            yield from super().modify_tensors(data_torch, name, bid)
            return
        m = re.match(re.escape(PREFIX) + r"(\d+)\.(.+)\.weight$", name)
        layer, hf = int(m[1]), m[2]
        vrole, first, n_rows = self.hf2vllm[hf]
        vkey = f"layers.{layer}.{vrole}"
        cols = self.sidecar.layers[vkey]["in_features"]
        emitted = []
        for gname, src_rows, col_perm in self._probe(name, bid, n_rows, cols):
            if hf == "self_attn.q_proj":
                # Qwen3.5 interleaves each head's query and output gate, [q | gate] per head. Mirai keeps the queries
                # and the gates in separate trellis blocks, sometimes of different formats: attn_q + attn_gate.
                is_gate = (src_rows // self.hparams["head_dim"]) % 2 == 1
                emitted += [(gname, src_rows[~is_gate], col_perm),
                            (gname.replace(".attn_q.", ".attn_gate."), src_rows[is_gate], col_perm)]
            else:
                emitted.append((gname, src_rows, col_perm))
        for gname, src_rows, col_perm in emitted:
            fmt, packets, entries, rowscale = self.sidecar.rows(vkey, first + src_rows)
            if col_perm:
                assert hf == "linear_attn.out_proj", f"{name}: unexpected column permutation"
            if len(src_rows) % 32 != 0:
                # 48-row gates (ssm_alpha / ssm_beta): decode to dense F16 in llama.cpp's row order
                w_rot = md.rotated_rows(fmt, packets, entries, rowscale, self.sidecar.get(f"codebook.{fmt[:2]}"))
                w = md.unrotate(w_rot, self.sidecar.get(f"rotation.signs_{cols}"), self.sidecar.get(f"rotation.q_{cols}"))
                assert not col_perm
                self._raw.append((gname, w.astype(np.float16), gguf.GGMLQuantizationType.F16))
                continue
            self._raw.append((gname, interleave(packets, entries), QTYPE[fmt]))
            self._raw.append((gname.replace(".weight", ".scale"), rowscale.astype(np.float32), gguf.GGMLQuantizationType.F32))
            if self.verify_dir is not None:
                self._verify(gname, name, fmt, packets, entries, rowscale, src_rows, cols)
        return
        yield  # noqa: unreachable, keeps this a generator

    def _verify(self, gname, hf_name, fmt, packets, entries, rowscale, src_rows, cols):
        """Correlate decoded rows with the base model's bf16 rows (HF order, so columns match the codes)."""
        k = min(self.verify_rows, len(src_rows))
        pick = np.linspace(0, len(src_rows) - 1, k).astype(np.int64)
        w_rot = md.rotated_rows(fmt, packets[pick], entries[pick], rowscale[pick],
                                self.sidecar.get(f"codebook.{fmt[:2]}"))
        w = md.unrotate(w_rot, self.sidecar.get(f"rotation.signs_{cols}"), self.sidecar.get(f"rotation.q_{cols}"))
        base = self._base_rows(hf_name, src_rows[pick])
        corr = [float(np.corrcoef(a, b)[0, 1]) for a, b in zip(w, base)]
        ratio = float(np.linalg.norm(w) / np.linalg.norm(base))
        rec = {"tensor": gname, "format": fmt, "rows": int(len(src_rows)), "corr_min": min(corr),
               "corr_mean": float(np.mean(corr)), "norm_ratio": ratio}
        self._verify_report.append(rec)
        logger.info("verify %s", json.dumps(rec))
        assert min(corr) > 0.8, f"{gname}: decoded rows do not match the base model ({rec})"

    def _base_rows(self, hf_name: str, rows: np.ndarray) -> np.ndarray:
        if not hasattr(self, "_base_index"):
            idx = json.load(open(self.verify_dir / "model.safetensors.index.json"))["weight_map"]
            self._base_index = idx
        hf_name = hf_name.replace(PREFIX, "model.language_model.layers.")
        f = safe_open(str(self.verify_dir / self._base_index[hf_name]), "pt")
        sl = f.get_slice(hf_name)
        return np.stack([sl[int(r):int(r) + 1].float().numpy()[0] for r in rows]).astype(np.float64)

    def prepare_tensors(self):
        super().prepare_tensors()
        sc = self.sidecar
        h = self.hparams
        n_embd = h["hidden_size"]
        # shared rotations and codebooks
        for cols in (5120, 6144, 17408):
            if f"rotation.signs_{cols}" in sc.f.keys():
                rot = np.concatenate([sc.get(f"rotation.signs_{cols}").astype(np.float32),
                                      sc.get(f"rotation.q_{cols}").astype(np.float32).reshape(-1)])
                self._raw.append((f"mirai.rot.{cols}", rot, gguf.GGMLQuantizationType.F32))
        for v in ("v4", "v2"):
            self.gguf_writer.add_array(f"mirai.codebook.{v}", [float(x) for x in sc.get(f"codebook.{v}")])
        self.gguf_writer.add_uint32("mirai.version", 1)
        # head: interleave codes and ladder nibbles per 32 rows, like the plugin's head_mma
        codes = sc.get("head.codes")
        ladder_idx = sc.get("head.ladder_indices")
        vocab, pairs = codes.shape[0], ladder_idx.shape[1]
        assert vocab % 32 == 0 and codes.shape[1] == pairs * 48
        blocks = vocab // 32  # per 32 rows: codes [pair][part][row][16 B], then ladder bytes [pair][row] (head_mma's order)
        head_codes = codes.reshape(blocks, 32, pairs, 3, 16).transpose(0, 2, 3, 1, 4).reshape(blocks, -1)
        head_ladder = ladder_idx.reshape(blocks, 32, pairs).transpose(0, 2, 1).reshape(blocks, -1)
        head = np.concatenate([head_codes, head_ladder], axis=1).reshape(vocab, -1)
        self._raw.append(("output.weight", head, gguf.GGMLQuantizationType.MS_I3))
        self._raw.append(("output.scale", sc.get("head.row_scales").astype(np.float32), gguf.GGMLQuantizationType.F32))
        aux = np.concatenate([sc.get("head.signs").astype(np.float32), sc.get("head.ladder").astype(np.float32)])
        self._raw.append(("mirai.head_aux", aux, gguf.GGMLQuantizationType.F32))
        # embedding: decode D4 rows (table[code][col % 4] * row_scale * ladder, H32 per 32 columns, signs) to F16
        self._raw.append(("token_embd.weight", self._decode_embedding(), gguf.GGMLQuantizationType.F16))
        for name, data, qtype in self._raw:
            logger.info("%-40s --> %s, shape = %s", name, qtype.name, data.shape)
            self.gguf_writer.add_tensor(name, data, raw_dtype=qtype)
        if self._verify_report:
            (self.fname_out.parent / (self.fname_out.stem + ".verify.json")).write_text(json.dumps(self._verify_report, indent=1))

    def _decode_embedding(self) -> np.ndarray:
        sc = self.sidecar
        codes = sc.get("embedding.codes")
        table = sc.get("embedding.table").astype(np.float32)
        row_scales = sc.get("embedding.row_scales").astype(np.float32)
        ladder = sc.get("embedding.ladder").astype(np.float32)
        ladder_idx = sc.get("embedding.ladder_indices")
        signs = sc.get("embedding.signs").astype(np.float32)
        vocab, n_embd = codes.shape[0], codes.shape[1] * 4
        out = np.empty((vocab, n_embd), np.float16)
        groups = n_embd // 64
        g = np.arange(groups)
        for start in range(0, vocab, 8192):
            end = min(vocab, start + 8192)
            c = codes[start:end]
            vals = table[c].reshape(end - start, n_embd)  # [rows, cols/4, 4] -> cols
            li = ladder_idx[start:end][:, g // 2]
            nib = np.where(g % 2 == 1, li >> 4, li & 15)
            scale = row_scales[start:end, None] * ladder[nib]  # [rows, groups]
            x = vals * np.repeat(scale, 64, axis=1)
            x = md.hadamard32_rows(x.astype(np.float64)) * signs[None, :]
            out[start:end] = x.astype(np.float16)
        return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", type=Path, help="the vllm/ folder: config.json, tokenizer, dense shards, trellis.mirai")
    ap.add_argument("--outfile", type=Path, required=True)
    ap.add_argument("--mtp-type", choices=["q8_0", "bf16", "f16"], default="q8_0", help="type of the dense MTP weights")
    ap.add_argument("--verify-base", type=Path, help="bf16 Qwen3.8-27B folder: correlate decoded rows with it")
    ap.add_argument("--verify-rows", type=int, default=8)
    ap.add_argument("--no-mtp", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)

    hparams = MiraiSQwen35.load_hparams(args.model, False)
    hparams.pop("quantization_config", None)  # the stock loader rejects unknown quant methods
    hparams["architectures"] = ["Qwen3_5ForCausalLM"]
    ftype = {"q8_0": gguf.LlamaFileType.MOSTLY_Q8_0, "bf16": gguf.LlamaFileType.MOSTLY_BF16,
             "f16": gguf.LlamaFileType.MOSTLY_F16}[args.mtp_type]
    if args.no_mtp:
        MiraiSQwen35.no_mtp = True
    model = MiraiSQwen35(args.model, ftype, args.outfile, hparams=hparams, eager=False,
                         sidecar=Sidecar(args.model / "trellis.mirai"), verify_dir=args.verify_base,
                         verify_rows=args.verify_rows)
    model.write()
    logger.info("wrote %s", args.outfile)


if __name__ == "__main__":
    main()
