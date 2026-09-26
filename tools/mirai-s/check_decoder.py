"""Cross-check mirai_s_decode against the lalamo package's own parametrization of the same weights.

Decodes the same rows of one layer from the vLLM sidecar (packets, computed codebook, rowscale) and from the lalamo
model.safetensors (tape, stored table, scale * gain, post gain; the smatvec extract_leaf decoder) and compares.
"""
import argparse
import json
import sys

import numpy as np
from safetensors import safe_open

import mirai_s_decode as md

ap = argparse.ArgumentParser()
ap.add_argument("--sidecar", required=True)
ap.add_argument("--lalamo-dir", required=True, help="dir holding smatvec's extract_leaf.py")
ap.add_argument("--vllm-key", default="layers.10.mlp.down_proj")
ap.add_argument("--lalamo-leaf", default="decoder.transformer.layers.10.mlp.down_projection.weights")
ap.add_argument("--rows", type=int, default=64)
args = ap.parse_args()

sys.path.insert(0, args.lalamo_dir)
import extract_leaf as el  # noqa: E402

f = safe_open(args.sidecar, "np")
meta = json.loads(f.metadata()["layers"])[args.vllm_key]
assert not meta["has_perm"] and len(meta["blocks"]) == 1, meta
fmt = meta["blocks"][0]["format"]
v, t, steps, words, edt = md.FORMATS[fmt]
cols = meta["in_features"]
p = cols // (steps * v)
packets = f.get_tensor(f"{args.vllm_key}.0.packets")
entries = f.get_tensor(f"{args.vllm_key}.0.entries").view(edt)
rowscale = f.get_tensor(f"{args.vllm_key}.0.rowscale")
g = rowscale.size // 32
packets = packets.reshape(g, p, words, 32, 16).transpose(0, 3, 1, 2, 4).reshape(g * 32, p, words, 16)
entries = entries.reshape(g, p, 32).transpose(0, 2, 1).reshape(g * 32, p)
n = args.rows
codebook = f.get_tensor(f"codebook.{fmt[:2]}")
w_rot = md.rotated_rows(fmt, packets[:n], entries[:n], rowscale[:n], codebook)
signs = f.get_tensor(f"rotation.signs_{cols}")
q = f.get_tensor(f"rotation.q_{cols}")
w_side = md.unrotate(w_rot, signs, q)

# lalamo: same rows, exact f64 decode of its own parametrization
pkg = el.Package(el.MODEL)
spec, shape = pkg.leaves()[args.lalamo_leaf]
codes = pkg.get(args.lalamo_leaf + ".codes")[:n]
st = el.states(spec, codes, cols)
table = pkg.get(f"qtip_shared.codebook_v{spec['vector_width']}").astype(np.float64)
vals = table[st].reshape(n, cols)
sc = pkg.get(args.lalamo_leaf + ".scales").astype(np.float64)[:n]
gn = pkg.get(args.lalamo_leaf + ".gains").astype(np.float64)[:n]
pg = pkg.get(args.lalamo_leaf + ".post_gains.0").astype(np.float64)[:n]
lal_rot = vals * sc[:, None] * gn[:, None] * pg[:, None]
w_lal = md.unrotate(lal_rot, pkg.get(f"qtip_shared.signs_{cols}"), pkg.get(f"qtip_shared.q_{cols}"))

rel_rot = np.linalg.norm(w_rot - lal_rot) / np.linalg.norm(lal_rot)
rel = np.linalg.norm(w_side - w_lal) / np.linalg.norm(w_lal)
print(json.dumps({"key": args.vllm_key, "format": fmt, "rows": n, "rel_l2_rotated": rel_rot, "rel_l2_weights": rel,
                  "max_abs": float(np.abs(w_side - w_lal).max()), "max_weight": float(np.abs(w_lal).max())}))
