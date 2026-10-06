#!/usr/bin/env python3
"""A refusal direction as a llama.cpp control vector, from residual dumps of llama-resid-dump.

    refusal_direction.py OUT.gguf --pair harmful.A.bin harmless.A.bin [--pair harmful.B.bin harmless.B.bin ...]
                         [--layers 1-63] [--model-hint qwen35]

Each --pair is one prompt context (for example thinking off, thinking on, with a system message): two dumps of the
same shape [prompts, layers, n_embd], made with

    llama-resid-dump -m model.gguf --positive-file prompts.txt -o harmful.A.bin -ngl 99 -c 2048

where prompts.txt holds one prompt per line, already in the chat template ("\\n" escapes), and the last token of the
line is where the state is read.

Per layer and context: d = mean(harmful) - mean(harmless); the component of d along the mean harmless state of that
layer is removed; d is normalized. The file holds the normalized mean over the contexts, one direction per layer.
Use it with --control-vector-scaled OUT.gguf:1.0 --cvec-mode project.

Why the component along the harmless mean is removed: an ordinary prompt can have a large part of its norm along the
plain difference of means (15% to 54% on Mirai S), and projecting that out breaks the model.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'gguf-py'))
import gguf  # noqa: E402


def load(path):
    meta = json.loads(Path(str(path) + '.json').read_text())
    a = np.fromfile(path, dtype=np.float32)
    return a.reshape(meta['n_prompts'], meta['n_layer'], meta['n_embd']).astype(np.float64)


def unit_directions(harmful, harmless):
    mu = load(harmless).mean(0)                                   # [layers, n_embd]
    d = load(harmful).mean(0) - mu
    m = mu / np.linalg.norm(mu, axis=-1, keepdims=True)
    d = d - (d * m).sum(-1, keepdims=True) * m
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('out')
    ap.add_argument('--pair', nargs=2, action='append', required=True, metavar=('HARMFUL', 'HARMLESS'))
    ap.add_argument('--layers', default=None, help='A-B, inclusive (default: 1 to the last layer; layer 0 cannot be steered)')
    ap.add_argument('--model-hint', default='qwen35')
    a = ap.parse_args()

    v = np.mean([unit_directions(h, s) for h, s in a.pair], axis=0)
    v /= np.linalg.norm(v, axis=-1, keepdims=True)
    n_layer = v.shape[0]
    first, last = map(int, a.layers.split('-')) if a.layers else (1, n_layer - 1)

    w = gguf.GGUFWriter(a.out, 'controlvector')
    w.add_string('controlvector.model_hint', a.model_hint)
    w.add_uint32('controlvector.layer_count', n_layer - 1)
    for layer in range(first, last + 1):
        w.add_tensor(f'direction.{layer}', v[layer].astype(np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    data = Path(a.out).read_bytes()
    print(json.dumps({'file': a.out, 'contexts': len(a.pair), 'layers': [first, last], 'bytes': len(data),
                      'sha256': hashlib.sha256(data).hexdigest()}))


if __name__ == '__main__':
    main()
