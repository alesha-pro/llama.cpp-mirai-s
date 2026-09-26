#include "gated_delta_net.cuh"
#include "ggml-cuda/common.cuh"

template <int S_v, bool KDA, bool keep_rs_t>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, 2)
gated_delta_net_cuda(const float * q,
                                     const float * k,
                                     const float * v,
                                     const float * g,
                                     const float * beta,
                                     const float * curr_state,
                                     float *       dst,
                                     float *       state,
                                     int64_t       H,
                                     int64_t       n_tokens,
                                     int64_t       n_seqs,
                                     int64_t       sq1,
                                     int64_t       sq2,
                                     int64_t       sq3,
                                     int64_t       sv1,
                                     int64_t       sv2,
                                     int64_t       sv3,
                                     int64_t       sb1,
                                     int64_t       sb2,
                                     int64_t       sb3,
                                     const uint3   neqk1_magic,
                                     const uint3   rq3_magic,
                                     float         scale,
                                     int64_t       state_slot_stride,
                                     int           K) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    // each warp owns one column, using warp-level primitives to reduce across rows
    const int      lane     = threadIdx.x;
    const int      col      = blockIdx.z * blockDim.y + threadIdx.y;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    float *       attn_data        = dst;

    // input state holds s0 only: [S_v, S_v, H, n_seqs] — seq stride is D = H * S_v * S_v.
    // output state layout (per-slot D * n_seqs) — same per-(seq,head) offset as before.
    const int64_t state_in_offset      = sequence * H * S_v * S_v + h_idx * S_v * S_v;
    const int64_t state_out_offset     = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    curr_state += state_in_offset + col * S_v;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr int rows_per_lane = (S_v + warp_size - 1) / warp_size;
    float         s_shard[rows_per_lane];
    // state is stored transposed: M[col][i] = S[i][col], row col is contiguous

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * warp_size + lane;
        s_shard[r]  = curr_state[i];
    }

    for (int t = 0; t < n_tokens; t++) {
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float * beta_t = beta + gb_offset;
        const float * g_t    = g    + gb_offset * (KDA ? S_v : 1);

        const float beta_val = *beta_t;

        // Cache k and q in registers
        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            k_reg[r] = k_t[i];
            q_reg[r] = q_t[i];
        }

        if constexpr (!KDA) {
            const float g_val = expf(*g_t);

            // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kv_shard += s_shard[r] * k_reg[r];
            }
            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - g * kv[col]) * beta
            float delta_col = (v_t[col] - g_val * kv_col) * beta_val;

            // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[r]  = g_val * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        } else {
            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                kv_shard += expf(g_t[i]) * s_shard[r] * k_reg[r];
            }

            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            float delta_col = (v_t[col] - kv_col) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                s_shard[r]  = expf(g_t[i]) * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        }

        attn_data += S_v * H;

        if constexpr (keep_rs_t) {
            // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
            // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < K) {
                float * curr_state = state + target_slot * state_slot_stride;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * warp_size + lane;
                    curr_state[col * S_v + i] = s_shard[r];
                }
            }
        }
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i          = r * warp_size + lane;
            state[col * S_v + i] = s_shard[r];
        }
    }
}

// Chunked form of the scalar-gate delta rule for prefill (FLA's chunk_gated_delta_rule, fp32 throughout).
// Within a chunk of C tokens, with D_t = exp(cumsum g) from the chunk start and S0 the state before it:
//   A[t][j] = beta_t D_t/D_j (k_t.k_j) for j < t
//   (I + A) Delta = beta (V - D K S0)                (Delta = the per-token deltas of the recurrence)
//   O[t]    = D_t S0^T q_t + sum_{j<=t} D_t/D_j (q_t.k_j) Delta[j]
//   S_C     = D_C S0 + sum_j D_C/D_j k_j Delta[j]^T
// One block per (head, sequence, VS-column slice of the state); the chunk loop is sequential. The triangular
// system is solved by forward substitution, 8 threads per column of Delta.
template <int S_v, int C, int VS>
__global__ void __launch_bounds__(256, 1)
gated_delta_net_chunked_cuda(const float * q, const float * k, const float * v, const float * g, const float * beta,
                             const float * curr_state, float * dst, float * state,
                             int64_t H, int64_t n_tokens,
                             int64_t sq1, int64_t sq2, int64_t sq3,
                             int64_t sv1, int64_t sv2, int64_t sv3,
                             int64_t sb1, int64_t sb2, int64_t sb3,
                             const uint3 neqk1_magic, const uint3 rq3_magic, float scale) {
    static_assert(C == 32 && VS == 32 && S_v % 32 == 0, "tile mapping assumes C == VS == 32");
    constexpr int SK  = S_v + 4;         // row stride of the C x S_v tiles (float4 aligned, fewer bank conflicts)
    constexpr int NT  = 256;
    constexpr int NKQ = C*(S_v/4) / NT;  // float4 of K (and of Q) per thread and chunk

    extern __shared__ float smem[];
    float * Ks  = smem;              // [C][SK]   keys
    float * Qs  = Ks  + C*SK;        // [C][SK]   queries
    float * As  = Qs  + C*SK;        // [C][C]    A (strictly lower)
    float * QKs = As  + C*C;         // [C][C]    D_t/D_j q_t.k_j (lower, with diagonal)
    float * UV  = QKs + C*C;         // [C][VS]   right-hand side, then Delta
    float * Ss  = UV  + C*VS;        // [S_v][VS] state slice S[i][c0 + c]
    float * gam = Ss  + S_v*VS;      // [C]       cumulative log decay
    float * bet = gam + C;           // [C]

    const int h        = blockIdx.x;
    const int sequence = blockIdx.y;
    const int c0       = blockIdx.z * VS;
    const int tid      = threadIdx.x;

    const uint32_t iq1 = fastmodulo(h, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    const float * q_s = q + iq3*sq3 + iq1*sq1;
    const float * k_s = k + iq3*sq3 + iq1*sq1;
    const float * v_s = v + sequence*sv3 + h*sv1 + c0;
    const float * g_s = g    + sequence*sb3 + h*sb1;
    const float * b_s = beta + sequence*sb3 + h*sb1;
    float * attn = dst + (sequence*n_tokens*H + h)*S_v + c0;

    const int64_t state_offset = (sequence*H + h)*S_v*S_v;
    for (int e = tid; e < S_v*VS; e += NT) {
        const int i = e / VS, c = e % VS;
        Ss[i*VS + c] = curr_state[state_offset + (c0 + c)*S_v + i];
    }

    // chunk data in registers, loaded one chunk ahead; rows past the end are zero (no contribution, no decay)
    float4 kr[NKQ], qr[NKQ];
    float  vr[C*VS/NT];
    float  gr = 0.0f, br = 0.0f;
    auto load_chunk = [&](int64_t t0) {
        const int nc = (int) min((int64_t) C, n_tokens - t0);
#pragma unroll
        for (int r = 0; r < NKQ; ++r) {
            const int e = tid + r*NT, i = e / (S_v/4), d4 = e % (S_v/4);
            kr[r] = i < nc ? *(const float4 *) (k_s + (t0 + i)*sq2 + 4*d4) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            qr[r] = i < nc ? *(const float4 *) (q_s + (t0 + i)*sq2 + 4*d4) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
#pragma unroll
        for (int r = 0; r < C*VS/NT; ++r) {
            const int e = tid + r*NT, i = e / VS, c = e % VS;
            vr[r] = i < nc ? v_s[(t0 + i)*sv2 + c] : 0.0f;
        }
        if (tid < C) {
            gr = tid < nc ? g_s[(t0 + tid)*sb2] : 0.0f;
            br = tid < nc ? b_s[(t0 + tid)*sb2] : 0.0f;
        }
    };
    load_chunk(0);

    // 2x2 register tiles over the 32x32 outputs: rows 2a, 2a+1, columns 2b, 2b+1
    const int ta = tid / 16;
    const int tb = tid % 16;

    for (int64_t t0 = 0; t0 < n_tokens; t0 += C) {
        const int nc = (int) min((int64_t) C, n_tokens - t0);

#pragma unroll
        for (int r = 0; r < NKQ; ++r) {
            const int e = tid + r*NT, i = e / (S_v/4), d4 = e % (S_v/4);
            *(float4 *) (Ks + i*SK + 4*d4) = kr[r];
            *(float4 *) (Qs + i*SK + 4*d4) = qr[r];
        }
#pragma unroll
        for (int r = 0; r < C*VS/NT; ++r) {
            UV[tid + r*NT] = vr[r];
        }
        if (tid < 32) {
            float gi = gr;
#pragma unroll
            for (int off = 1; off < 32; off <<= 1) {
                const float o = __shfl_up_sync(0xFFFFFFFF, gi, off);
                if (tid >= off) {
                    gi += o;
                }
            }
            gam[tid] = gi;
            bet[tid] = br;
        }
        __syncthreads();

        if (t0 + C < n_tokens) {
            load_chunk(t0 + C);
        }

        // one pass over the head dimension: A, QK and the (i, c) tiles of K S0 and Q S0
        float aa[2][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};
        float qk[2][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};
        float ks[2][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};
        float qs[2][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};
#pragma unroll 2
        for (int d = 0; d < S_v; d += 4) {
            const float4 k0 = *(const float4 *) (Ks + (2*ta + 0)*SK + d);
            const float4 k1 = *(const float4 *) (Ks + (2*ta + 1)*SK + d);
            const float4 q0 = *(const float4 *) (Qs + (2*ta + 0)*SK + d);
            const float4 q1 = *(const float4 *) (Qs + (2*ta + 1)*SK + d);
            const float4 j0 = *(const float4 *) (Ks + (2*tb + 0)*SK + d);
            const float4 j1 = *(const float4 *) (Ks + (2*tb + 1)*SK + d);
            aa[0][0] += k0.x*j0.x + k0.y*j0.y + k0.z*j0.z + k0.w*j0.w;
            aa[0][1] += k0.x*j1.x + k0.y*j1.y + k0.z*j1.z + k0.w*j1.w;
            aa[1][0] += k1.x*j0.x + k1.y*j0.y + k1.z*j0.z + k1.w*j0.w;
            aa[1][1] += k1.x*j1.x + k1.y*j1.y + k1.z*j1.z + k1.w*j1.w;
            qk[0][0] += q0.x*j0.x + q0.y*j0.y + q0.z*j0.z + q0.w*j0.w;
            qk[0][1] += q0.x*j1.x + q0.y*j1.y + q0.z*j1.z + q0.w*j1.w;
            qk[1][0] += q1.x*j0.x + q1.y*j0.y + q1.z*j0.z + q1.w*j0.w;
            qk[1][1] += q1.x*j1.x + q1.y*j1.y + q1.z*j1.z + q1.w*j1.w;
            const float k0a[4] = {k0.x, k0.y, k0.z, k0.w};
            const float k1a[4] = {k1.x, k1.y, k1.z, k1.w};
            const float q0a[4] = {q0.x, q0.y, q0.z, q0.w};
            const float q1a[4] = {q1.x, q1.y, q1.z, q1.w};
#pragma unroll
            for (int dd = 0; dd < 4; ++dd) {
                const float2 sv = *(const float2 *) (Ss + (d + dd)*VS + 2*tb);
                ks[0][0] += k0a[dd]*sv.x; ks[0][1] += k0a[dd]*sv.y;
                ks[1][0] += k1a[dd]*sv.x; ks[1][1] += k1a[dd]*sv.y;
                qs[0][0] += q0a[dd]*sv.x; qs[0][1] += q0a[dd]*sv.y;
                qs[1][0] += q1a[dd]*sv.x; qs[1][1] += q1a[dd]*sv.y;
            }
        }
#pragma unroll
        for (int ii = 0; ii < 2; ++ii) {
            const int i = 2*ta + ii;
            const float gi = gam[i];
            const float bi = bet[i];
            const float di = expf(gi);
#pragma unroll
            for (int jj = 0; jj < 2; ++jj) {
                const int j = 2*tb + jj;
                const float f = expf(gi - gam[j]);
                As [i*C + j] = i >  j ? bi * f * aa[ii][jj] : 0.0f;
                QKs[i*C + j] = i >= j ?      f * qk[ii][jj] : 0.0f;
            }
            // right-hand side beta (V - D K S0) in place of V; D Q S0 stays in registers for the output
            float2 * u = (float2 *) (UV + i*VS + 2*tb);
            float2 uv = *u;
            uv.x = bi * (uv.x - di*ks[ii][0]);
            uv.y = bi * (uv.y - di*ks[ii][1]);
            *u = uv;
            qs[ii][0] *= di;
            qs[ii][1] *= di;
        }
        __syncthreads();

        // (I + A) Delta = rhs: column c = tid / 8, lane l = tid % 8 holds rows l, l + 8, l + 16, l + 24
        {
            const int c = tid / 8;
            const int l = tid % 8;
            float x[C/8];
#pragma unroll
            for (int r = 0; r < C/8; ++r) {
                x[r] = UV[(l + 8*r)*VS + c];
            }
#pragma unroll
            for (int i = 1; i < C; ++i) {
                float part = 0.0f;
#pragma unroll
                for (int r = 0; r < C/8; ++r) {
                    const int j = l + 8*r;
                    if (j < i) {
                        part += As[i*C + j] * x[r];
                    }
                }
                part += __shfl_xor_sync(0xFFFFFFFF, part, 1, 8);
                part += __shfl_xor_sync(0xFFFFFFFF, part, 2, 8);
                part += __shfl_xor_sync(0xFFFFFFFF, part, 4, 8);
                if (l == i % 8) {
                    x[i / 8] -= part;
                }
            }
#pragma unroll
            for (int r = 0; r < C/8; ++r) {
                UV[(l + 8*r)*VS + c] = x[r];
            }
        }
        __syncthreads();

        // O = D Q S0 + QK Delta
        {
#pragma unroll 8
            for (int j = 0; j < C; ++j) {
                const float2 dv = *(const float2 *) (UV + j*VS + 2*tb);
                const float a0 = QKs[(2*ta + 0)*C + j];
                const float a1 = QKs[(2*ta + 1)*C + j];
                qs[0][0] += a0*dv.x; qs[0][1] += a0*dv.y;
                qs[1][0] += a1*dv.x; qs[1][1] += a1*dv.y;
            }
#pragma unroll
            for (int ii = 0; ii < 2; ++ii) {
                const int i = 2*ta + ii;
                if (i < nc) {
                    float * o = attn + (t0 + i)*S_v*H + 2*tb;
                    o[0] = qs[ii][0] * scale;
                    o[1] = qs[ii][1] * scale;
                }
            }
        }

        // S = D_C S0 + (D_C/D K)^T Delta; each thread owns a 4x4 tile of the slice
        if (tid < S_v*VS/16) {
            const float gl = gam[C - 1];
            const int d0 = 4*(tid / 8);
            const int cc = 4*(tid % 8);
            float acc[4][4];
            const float dc = expf(gl);
#pragma unroll
            for (int ii = 0; ii < 4; ++ii) {
                const float4 sv = *(const float4 *) (Ss + (d0 + ii)*VS + cc);
                acc[ii][0] = dc*sv.x; acc[ii][1] = dc*sv.y; acc[ii][2] = dc*sv.z; acc[ii][3] = dc*sv.w;
            }
#pragma unroll 4
            for (int j = 0; j < C; ++j) {
                const float f = expf(gl - gam[j]);
                const float4 kv = *(const float4 *) (Ks + j*SK + d0);
                const float4 dv = *(const float4 *) (UV + j*VS + cc);
                const float ka[4] = {f*kv.x, f*kv.y, f*kv.z, f*kv.w};
#pragma unroll
                for (int ii = 0; ii < 4; ++ii) {
                    acc[ii][0] += ka[ii]*dv.x; acc[ii][1] += ka[ii]*dv.y;
                    acc[ii][2] += ka[ii]*dv.z; acc[ii][3] += ka[ii]*dv.w;
                }
            }
#pragma unroll
            for (int ii = 0; ii < 4; ++ii) {
                *(float4 *) (Ss + (d0 + ii)*VS + cc) = make_float4(acc[ii][0], acc[ii][1], acc[ii][2], acc[ii][3]);
            }
        }
        __syncthreads();
    }

    float * st = state + state_offset;
    for (int e = tid; e < S_v*VS; e += NT) {
        const int i = e / VS, c = e % VS;
        st[(c0 + c)*S_v + i] = Ss[i*VS + c];
    }
}

// GGML_CUDA_GDN_CHUNKED=0 keeps the token-by-token kernel for prefill too
static bool ggml_cuda_gdn_chunked_enabled() {
    static const bool enabled = [] {
        const char * env = getenv("GGML_CUDA_GDN_CHUNKED");
        return env == nullptr || atoi(env) != 0;
    }();
    return enabled;
}

template <int S_v>
static void launch_gated_delta_net_chunked(
        const float * q_d, const float * k_d, const float * v_d, const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d, int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1, int64_t sq2, int64_t sq3, int64_t sv1, int64_t sv2, int64_t sv3,
        int64_t sb1, int64_t sb2, int64_t sb3, int64_t neqk1, int64_t rq3, float scale, cudaStream_t stream) {
    constexpr int C  = 32;
    constexpr int VS = 32;
    constexpr size_t nbytes_shared = (2*C*(S_v + 4) + 2*C*C + C*VS + S_v*VS + 2*C) * sizeof(float);
    static bool attr_set[GGML_CUDA_MAX_DEVICES] = {false};
    const int dev = ggml_cuda_get_device();
    if (!attr_set[dev]) {
        CUDA_CHECK(cudaFuncSetAttribute(gated_delta_net_chunked_cuda<S_v, C, VS>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared));
        attr_set[dev] = true;
    }
    const dim3 grid_dims(H, n_seqs, S_v / VS);
    const dim3 block_dims(256, 1, 1);
    gated_delta_net_chunked_cuda<S_v, C, VS><<<grid_dims, block_dims, nbytes_shared, stream>>>(
        q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3,
        init_fastdiv_values(neqk1), init_fastdiv_values(rq3), scale);
}

template <bool KDA, bool keep_rs_t>
static void launch_gated_delta_net(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K, cudaStream_t stream) {
    //TODO: Add chunked kernel for even faster pre-fill
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int num_warps = 4;
    dim3      grid_dims(H, n_seqs, (S_v + num_warps - 1) / num_warps);
    dim3      block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
    switch (S_v) {
        case 16:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<16, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        case 32:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<32, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        case 64: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<64, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        }
        case 128: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<128, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        }
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

static void ggml_cuda_op_gated_delta_net_impl(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_cuda_gated_delta_net_fused_cache * cache) {
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;

    const int64_t rq3 = nev3 / neq3;

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;

    const float * s_d   = (const float *) src_state->data;
    float *       dst_d = (float *) dst->data;

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    // strides in floats (beta strides used for both g and beta offset computation)
    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    cudaStream_t stream = ctx.stream();

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    const int K = ggml_get_op_params_i32(dst, 0);
    const bool keep_rs = K > 1;

    // recurrent state -> gdn_out tail (after attention scores), or the cache when fusing
    float * state_d           = dst_d + S_v * H * n_tokens * n_seqs;
    int64_t state_slot_stride = S_v * S_v * H * n_seqs;
    if (cache != nullptr) {
        state_d           = cache->data;
        state_slot_stride = cache->slot_stride;
    }

    // prefill: chunked kernel (scalar gate, aligned rows). With rollback snapshots (keep_rs) it runs all but the
    // last K tokens and the token-by-token kernel does the tail, writing the K snapshots (one sequence only).
    const bool chunked = !kda && n_tokens >= 64 && (S_v == 64 || S_v == 128) &&
        sq2 % 4 == 0 && sq1 % 4 == 0 && sq3 % 4 == 0 && (uintptr_t) q_d % 16 == 0 && (uintptr_t) k_d % 16 == 0 &&
        (!keep_rs || (n_seqs == 1 && n_tokens >= 64 + K)) && ggml_cuda_gdn_chunked_enabled();
    if (chunked) {
        const int64_t n1 = keep_rs ? n_tokens - K : n_tokens;
        ggml_cuda_pool_alloc<float> s_mid(ctx.pool());
        float * s1 = keep_rs ? s_mid.alloc(S_v*S_v*H) : state_d;
        if (S_v == 128) {
            launch_gated_delta_net_chunked<128>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, s1, H, n1, n_seqs,
                sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1, rq3, scale, stream);
        } else {
            launch_gated_delta_net_chunked<64>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, s1, H, n1, n_seqs,
                sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1, rq3, scale, stream);
        }
        if (keep_rs) {
            launch_gated_delta_net<false, true>(q_d + n1*sq2, k_d + n1*sq2, v_d + n1*sv2, g_d + n1*sb2, b_d + n1*sb2,
                s1, dst_d + n1*S_v*H, state_d, S_v, H, K, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        }
        return;
    }

    if (kda) {
        if (keep_rs) {
            launch_gated_delta_net<true, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        } else {
            launch_gated_delta_net<true, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        }
    } else {
        if (keep_rs) {
            launch_gated_delta_net<false, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        } else {
            launch_gated_delta_net<false, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        }
    }
}

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, nullptr);
}

void ggml_cuda_op_gated_delta_net_fused_cache(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_cuda_gated_delta_net_fused_cache cache) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, &cache);
}
