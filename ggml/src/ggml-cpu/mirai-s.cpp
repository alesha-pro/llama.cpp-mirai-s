// Mirai S linears on the CPU: the reference for the CUDA kernels (ggml-cuda/mirai-s.cu), with the same math and layouts.
//
// A trellis row stores, per packet of STEPS * V columns, a 16-bit entry state and STEPS symbols of T bits, LSB-first.
// Each step replays state = (state << T | symbol) & 0xFFFF, and the codebook is computed from the state:
//   h = fmix32(state * 0xCFCCB83F + 0x584B4AA3),  level(byte b of h) = 8 * pairs(b) + ((3 * (b & 15)) & 15) - 54
//   w_rot[col] = c * level[col] + d[col % 4]       (V4: bytes 0-3 of one step's hash, V2: bytes 0-1 of two steps)
// With x_rot = R (signs * x) quantized per token as s * (q0 + q1 / 254), every output is finished from two exact
// integer dot products over level + 54:
//   y = scale * (c * s * ((coarse - 54 sum q0) + (fine - 54 sum q1) / 254) + sum_r d_r S_r)
// where S_r = s * (sum q0 + sum q1 / 254) over the columns = r mod 4.
//
// Tensor layouts (tools/mirai-s/convert_mirai_s_to_gguf.py):
//   trellis W: groups of 32 rows, each group = packets [P][WORDS][32 rows][16 B], then entry states [P][32 rows]
//   MS_I3 W:   groups of 32 rows, each group = codes [K / 128][3][32 rows][16 B], then ladder bytes [K / 128][32 rows]
//   xq (I32 [K / 2 + 8, T]), read as one blob: q words [T][K / 4][2] (plane 0, plane 1 of four columns), then
//   stats [T][8] = {s, sum q0, sum q1, S_0 .. S_3, 0}

#include "ops.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"
#include "simd-mappings.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace {

struct ms_format {
    int v, t, steps, words, entry_bytes;
};

ms_format ms_format_of(ggml_type type) {
    switch (type) {
        case GGML_TYPE_MS_V4T8: return {4, 8, 16, 1, 1};
        case GGML_TYPE_MS_V2T4: return {2, 4, 32, 1, 2};
        case GGML_TYPE_MS_V2T6: return {2, 6, 64, 3, 2};
        default: GGML_ABORT("not a Mirai S trellis type");
    }
}

inline uint32_t fmix_hash(uint32_t state) {
    uint32_t x = state * 0xCFCCB83Fu + 0x584B4AA3u;
    x ^= x >> 16;
    x *= 0x85EBCA6Bu;
    return x ^ (x >> 16);
}

// level(b) + 54 for each byte of h, in [0, 111]
inline uint32_t levels_plus_54(uint32_t h) {
    const uint32_t nibble_pairs = (h & 0x33333333u) + ((h >> 2) & 0x33333333u);
    const uint32_t pairs = (nibble_pairs + (nibble_pairs >> 4)) & 0x0F0F0F0Fu;
    return (pairs << 3) + (((h & 0x0F0F0F0Fu) * 3u) & 0x0F0F0F0Fu);
}

inline uint32_t symbol_at(const uint32_t * bits, int index, int t) {
    const int bit = index * t, word = bit / 32, shift = bit % 32;
    uint64_t joined = bits[word];
    if (shift + t > 32) {
        joined |= static_cast<uint64_t>(bits[word + 1]) << 32;
    }
    return static_cast<uint32_t>(joined >> shift) & ((1u << t) - 1);
}

// one packet of one row -> level + 54 per column, as 32-bit words of four columns
void decode_packet(const ms_format & f, const uint32_t * bits, uint32_t state, uint32_t * out) {
    if (f.v == 4) {
        for (int step = 0; step < f.steps; ++step) {
            state = ((state << f.t) | symbol_at(bits, step, f.t)) & 0xFFFFu;
            out[step] = levels_plus_54(fmix_hash(state));
        }
    } else {
        for (int step = 0; step < f.steps; step += 2) {
            state = ((state << f.t) | symbol_at(bits, step, f.t)) & 0xFFFFu;
            const uint32_t first = fmix_hash(state);
            state = ((state << f.t) | symbol_at(bits, step + 1, f.t)) & 0xFFFFu;
            const uint32_t second = fmix_hash(state);
            out[step / 2] = levels_plus_54((first & 0xFFFFu) | (second << 16));
        }
    }
}

inline int dot4_us(uint32_t levels, uint32_t q) {
    int sum = 0;
    for (int b = 0; b < 4; ++b) {
        sum += static_cast<int>((levels >> (8 * b)) & 0xFFu) * static_cast<int>(static_cast<int8_t>((q >> (8 * b)) & 0xFFu));
    }
    return sum;
}

// in-place Walsh-Hadamard, strides 1, 2, 4, ... (the CUDA kernels' butterfly order)
void fwht(float * a, int n) {
    for (int stride = 1; stride < n; stride <<= 1) {
        for (int low = 0; low < n; low += 2 * stride) {
            for (int i = low; i < low + stride; ++i) {
                const float x = a[i], y = a[i + stride];
                a[i] = x + y;
                a[i + stride] = x - y;
            }
        }
    }
}

void quantize_trellis(const ggml_compute_params * params, ggml_tensor * dst) {
    const ggml_tensor * x = dst->src[0];
    const ggml_tensor * rot = dst->src[1];
    const int order = ggml_get_op_params_i32(dst, 1);
    const int64_t K = x->ne[0], T = ggml_nrows(x), width = K / order;
    const float * signs = static_cast<const float *>(rot->data);
    const float * small_q = signs + K;
    uint32_t * q = static_cast<uint32_t *>(dst->data);
    float * stats = reinterpret_cast<float *>(q + T * (K / 2));
    const float normalization = 1.0f / std::sqrt(static_cast<float>(width));

    std::vector<float> signed_x(K), column(width), rotated(K);
    for (int64_t token = params->ith; token < T; token += params->nth) {
        const float * xt = static_cast<const float *>(x->data) + token * K;
        for (int64_t i = 0; i < K; ++i) {
            signed_x[i] = xt[i] * signs[i];
        }
        float maximum = 0.0f;
        for (int o = 0; o < order; ++o) {
            for (int64_t w = 0; w < width; ++w) {
                float value = 0.0f;
                for (int c = 0; c < order; ++c) {
                    value += signed_x[w * order + c] * small_q[o * order + c];
                }
                column[w] = value;
            }
            fwht(column.data(), static_cast<int>(width));
            for (int64_t w = 0; w < width; ++w) {
                const float value = column[w] * normalization;
                rotated[w * order + o] = value;
                maximum = std::max(maximum, std::fabs(value));
            }
        }
        const float step = maximum > 0.0f ? maximum / 127.0f : 1.0f, inverse = 1.0f / step;
        int coarse_sum[4] = {}, fine_sum[4] = {};
        uint32_t * qt = q + token * (K / 2);
        for (int64_t g = 0; g < K / 4; ++g) {
            uint32_t plane0 = 0, plane1 = 0;
            for (int r = 0; r < 4; ++r) {
                const float scaled = rotated[4 * g + r] * inverse;
                const int coarse = std::min(127, std::max(-127, static_cast<int>(std::nearbyint(scaled))));
                const int fine = std::min(127, std::max(-127, static_cast<int>(std::nearbyint((scaled - coarse) * 254.0f))));
                plane0 |= (static_cast<uint32_t>(coarse) & 0xFFu) << (8 * r);
                plane1 |= (static_cast<uint32_t>(fine) & 0xFFu) << (8 * r);
                coarse_sum[r] += coarse;
                fine_sum[r] += fine;
            }
            qt[2 * g] = plane0;
            qt[2 * g + 1] = plane1;
        }
        float * st = stats + token * 8;
        st[0] = step;
        st[1] = static_cast<float>(coarse_sum[0] + coarse_sum[1] + coarse_sum[2] + coarse_sum[3]);
        st[2] = static_cast<float>(fine_sum[0] + fine_sum[1] + fine_sum[2] + fine_sum[3]);
        for (int r = 0; r < 4; ++r) {
            st[3 + r] = step * (static_cast<float>(coarse_sum[r]) + static_cast<float>(fine_sum[r]) / 254.0f);
        }
        st[7] = 0.0f;
    }
}

// x_rot = H32(signs * x) as f16
void quantize_head(const ggml_compute_params * params, ggml_tensor * dst) {
    const ggml_tensor * x = dst->src[0];
    const float * signs = static_cast<const float *>(dst->src[1]->data);
    const int64_t K = x->ne[0], T = ggml_nrows(x);
    const float normalization = 1.0f / std::sqrt(32.0f);
    float block[32];
    for (int64_t token = params->ith; token < T; token += params->nth) {
        const float * xt = static_cast<const float *>(x->data) + token * K;
        ggml_fp16_t * out = static_cast<ggml_fp16_t *>(dst->data) + token * K;
        for (int64_t b = 0; b < K; b += 32) {
            for (int i = 0; i < 32; ++i) {
                block[i] = xt[b + i] * signs[b + i];
            }
            fwht(block, 32);
            for (int i = 0; i < 32; ++i) {
                out[b + i] = GGML_CPU_FP32_TO_FP16(block[i] * normalization);
            }
        }
    }
}

void mul_mat_trellis(const ggml_compute_params * params, ggml_tensor * dst) {
    const ggml_tensor * w = dst->src[0];
    const ggml_tensor * xq = dst->src[1];
    const float * scale = static_cast<const float *>(dst->src[2]->data);
    const float * codebook = reinterpret_cast<const float *>(dst->op_params);
    const ms_format f = ms_format_of(w->type);
    const int64_t K = w->ne[0], N = w->ne[1], T = xq->ne[1];
    const int columns_per_packet = f.steps * f.v, words_per_packet = columns_per_packet / 4;
    const int64_t P = K / columns_per_packet;
    const size_t group_bytes = 32 * ggml_row_size(w->type, K), entry_offset = static_cast<size_t>(P) * f.words * 512;
    const uint32_t * q = static_cast<const uint32_t *>(xq->data);
    const float * stats = reinterpret_cast<const float *>(q + T * (K / 2));

    std::vector<int> sums(2 * T);
    uint32_t bits[4 * 3 + 1], levels[32];
    for (int64_t group = params->ith; group < N / 32; group += params->nth) {
        const uint8_t * base = static_cast<const uint8_t *>(w->data) + group * group_bytes;
        for (int lane = 0; lane < 32; ++lane) {
            std::fill(sums.begin(), sums.end(), 0);
            for (int64_t p = 0; p < P; ++p) {
                for (int word = 0; word < f.words; ++word) {
                    memcpy(bits + 4 * word, base + ((p * f.words + word) * 32 + lane) * 16, 16);
                }
                bits[4 * f.words] = 0;
                const uint8_t * entry = base + entry_offset + (p * 32 + lane) * f.entry_bytes;
                const uint32_t state = f.entry_bytes == 1 ? entry[0] : entry[0] | (static_cast<uint32_t>(entry[1]) << 8);
                decode_packet(f, bits, state, levels);
                for (int64_t token = 0; token < T; ++token) {
                    const uint32_t * qt = q + token * (K / 2) + p * words_per_packet * 2;
                    int coarse = 0, fine = 0;
                    for (int g = 0; g < words_per_packet; ++g) {
                        coarse += dot4_us(levels[g], qt[2 * g]);
                        fine += dot4_us(levels[g], qt[2 * g + 1]);
                    }
                    sums[2 * token] += coarse;
                    sums[2 * token + 1] += fine;
                }
            }
            const int64_t row = group * 32 + lane;
            for (int64_t token = 0; token < T; ++token) {
                const float * st = stats + token * 8;
                const int coarse = sums[2 * token] - 54 * static_cast<int>(std::nearbyint(st[1]));
                const int fine = sums[2 * token + 1] - 54 * static_cast<int>(std::nearbyint(st[2]));
                const float offsets = codebook[1] * st[3] + codebook[2] * st[4] + codebook[3] * st[5] + codebook[4] * st[6];
                const float dot = codebook[0] * st[0] * (static_cast<float>(coarse) + static_cast<float>(fine) * (1.0f / 254.0f));
                static_cast<float *>(dst->data)[token * N + row] = scale[row] * (dot + offsets);
            }
        }
    }
}

// logits[v] = row_scale[v] * sum_j (2c - 7) * ladder[group] * x_rot[j], the weight rounded to f16 once (as on CUDA)
void mul_mat_head(const ggml_compute_params * params, ggml_tensor * dst) {
    const ggml_tensor * w = dst->src[0];
    const ggml_tensor * xr = dst->src[1];
    const float * scale = static_cast<const float *>(dst->src[2]->data);
    const float * ladder = static_cast<const float *>(dst->src[3]->data);
    const int64_t K = w->ne[0], N = w->ne[1], T = xr->ne[1], pairs = K / 128;
    const size_t group_bytes = 32 * ggml_row_size(w->type, K), ladder_offset = static_cast<size_t>(pairs) * 3 * 32 * 16;

    float steps[16];
    for (int i = 0; i < 16; ++i) {
        steps[i] = GGML_CPU_FP16_TO_FP32(GGML_CPU_FP32_TO_FP16(ladder[i]));
    }
    std::vector<float> x(T * K), row(K);
    for (int64_t i = 0; i < T * K; ++i) {
        x[i] = GGML_CPU_FP16_TO_FP32(static_cast<const ggml_fp16_t *>(xr->data)[i]);
    }
    for (int64_t group = params->ith; group < N / 32; group += params->nth) {
        const uint8_t * base = static_cast<const uint8_t *>(w->data) + group * group_bytes;
        for (int lane = 0; lane < 32; ++lane) {
            for (int64_t pair = 0; pair < pairs; ++pair) {
                uint32_t words[12];
                for (int part = 0; part < 3; ++part) {
                    memcpy(words + 4 * part, base + ((pair * 3 + part) * 32 + lane) * 16, 16);
                }
                const uint8_t ladder_byte = base[ladder_offset + pair * 32 + lane];
                for (int half = 0; half < 2; ++half) {
                    const float step = steps[half ? ladder_byte >> 4 : ladder_byte & 15];
                    for (int j = 0; j < 64; ++j) {
                        const int bit = 192 * half + 3 * j, word = bit / 32, shift = bit % 32;
                        uint64_t joined = words[word];
                        if (word + 1 < 12) {
                            joined |= static_cast<uint64_t>(words[word + 1]) << 32;
                        }
                        const int c = static_cast<int>((joined >> shift) & 7u);
                        row[pair * 128 + half * 64 + j] = GGML_CPU_FP16_TO_FP32(GGML_CPU_FP32_TO_FP16((2 * c - 7) * step));
                    }
                }
            }
            const int64_t r = group * 32 + lane;
            for (int64_t token = 0; token < T; ++token) {
                const float * xt = x.data() + token * K;
                float acc = 0.0f;
                for (int64_t j = 0; j < K; ++j) {
                    acc += row[j] * xt[j];
                }
                static_cast<float *>(dst->data)[token * N + r] = acc * scale[r];
            }
        }
    }
}

} // namespace

void ggml_compute_forward_mirai_quantize(const ggml_compute_params * params, ggml_tensor * dst) {
    if (ggml_get_op_params_i32(dst, 0)) {
        quantize_head(params, dst);
    } else {
        quantize_trellis(params, dst);
    }
}

void ggml_compute_forward_mirai_mul_mat(const ggml_compute_params * params, ggml_tensor * dst) {
    if (dst->src[0]->type == GGML_TYPE_MS_I3) {
        mul_mat_head(params, dst);
    } else {
        mul_mat_trellis(params, dst);
    }
}
