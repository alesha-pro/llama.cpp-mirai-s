"""NumPy reference decoder for Mirai S trellis tapes (the layout of Mirai's vLLM plugin, kernels.cu).

A trellis row stores, per packet of STEPS * V columns, an entry state and STEPS symbols of T bits packed LSB-first.
Each step replays state = (state << T | symbol) & 0xFFFF; the codebook is computed from the state:
    h = fmix32(state * 0xCFCCB83F + 0x584B4AA3),  level(byte b of h) = 8 * pairs(b) + ((3 * (b & 15)) & 15) - 54
    w_rot[col] = rowscale * (c * level[col] + d[col % 4])
V4 uses bytes 0-3 of one step's hash, V2 bytes 0-1. The weight row itself is W = signs * (w_rot @ R), with
R = kron(H_width, Q) / sqrt(width) applied on the [width, order] view of the row (lalamo's full_rotation).
"""
import numpy as np

# format -> (V, T, steps per packet, 16-byte words per packet, entry dtype)
FORMATS = {
    "v4t8": (4, 8, 16, 1, np.uint8),
    "v2t4": (2, 4, 32, 1, np.uint16),
    "v2t6": (2, 6, 64, 3, np.uint16),
}
M32 = np.uint64(0xFFFFFFFF)


def fmix(state: np.ndarray) -> np.ndarray:
    x = (state.astype(np.uint64) * np.uint64(0xCFCCB83F) + np.uint64(0x584B4AA3)) & M32
    x ^= x >> np.uint64(16)
    x = (x * np.uint64(0x85EBCA6B)) & M32
    return (x ^ (x >> np.uint64(16))) & M32


def levels_plus_54(h: np.ndarray) -> np.ndarray:
    """Four bytes of level + 54 (0..111) per hash, as [..., 4] uint8."""
    h = h.astype(np.uint64)
    nib = (h & np.uint64(0x33333333)) + ((h >> np.uint64(2)) & np.uint64(0x33333333))
    pairs = (nib + (nib >> np.uint64(4))) & np.uint64(0x0F0F0F0F)
    lv = ((pairs << np.uint64(3)) + (((h & np.uint64(0x0F0F0F0F)) * np.uint64(3)) & np.uint64(0x0F0F0F0F))) & M32
    return np.stack([((lv >> np.uint64(8 * j)) & np.uint64(0xFF)).astype(np.uint8) for j in range(4)], axis=-1)


def row_levels(fmt: str, packets: np.ndarray, entries: np.ndarray) -> np.ndarray:
    """packets [n, P, W, 16] uint8 and entries [n, P] -> level + 54 per column, [n, P * STEPS * V] uint8."""
    v, t, steps, words, _ = FORMATS[fmt]
    n, p = entries.shape
    bits = packets.reshape(n, p, words * 16).view(np.uint32).astype(np.uint64)  # [n, P, 4 * W] little-endian words
    padded = np.concatenate([bits, np.zeros((n, p, 1), np.uint64)], axis=-1)
    mask = np.uint64((1 << t) - 1)
    state = entries.astype(np.uint64)
    out = np.empty((n, p, steps * v), np.uint8)
    for step in range(steps):
        bit = step * t
        word, shift = bit // 32, bit % 32
        joined = padded[..., word] | (padded[..., word + 1] << np.uint64(32))
        symbol = (joined >> np.uint64(shift)) & mask
        state = ((state << np.uint64(t)) | symbol) & np.uint64(0xFFFF)
        lv = levels_plus_54(fmix(state))
        out[..., step * v:(step + 1) * v] = lv[..., :v]
    return out.reshape(n, p * steps * v)


def rotated_rows(fmt: str, packets: np.ndarray, entries: np.ndarray, rowscale: np.ndarray,
                 codebook: np.ndarray) -> np.ndarray:
    """w_rot [n, K] float64: rowscale * (c * level + d[col % 4])."""
    lv = row_levels(fmt, packets, entries).astype(np.float64) - 54.0
    cols = lv.shape[1]
    d = np.asarray(codebook[1:5], np.float64)[np.arange(cols) % 4]
    return rowscale.astype(np.float64)[:, None] * (float(codebook[0]) * lv + d[None, :])


def fwht_rows(a: np.ndarray) -> np.ndarray:
    """Unnormalized Walsh-Hadamard on axis -2 of [..., width, order] (lalamo's butterfly order)."""
    width = a.shape[-2]
    stride = 1
    while stride < width:
        g = a.reshape(*a.shape[:-2], -1, 2, stride, a.shape[-1])
        left, right = g[..., 0, :, :], g[..., 1, :, :]
        a = np.concatenate((left + right, left - right), axis=-2).reshape(a.shape)
        stride *= 2
    return a


def unrotate(w_rot: np.ndarray, signs: np.ndarray, small_q: np.ndarray) -> np.ndarray:
    """W = signs * (w_rot @ kron(H, Q)) / sqrt(width), rows of w_rot in the [width, order] layout."""
    n, cols = w_rot.shape
    order = small_q.shape[0]
    width = cols // order
    a = fwht_rows(w_rot.reshape(n, width, order)) / np.sqrt(width)
    a = a @ small_q.astype(np.float64)
    return a.reshape(n, cols) * signs.astype(np.float64)[None, :]


def hadamard32_rows(x: np.ndarray) -> np.ndarray:
    """Normalized Walsh-Hadamard on consecutive 32-column blocks of [n, K]."""
    n, cols = x.shape
    return fwht_rows(x.reshape(n, cols // 32, 32, 1).transpose(0, 1, 2, 3).reshape(n * cols // 32, 32, 1)) \
        .reshape(n, cols) / np.sqrt(32.0)
