#include "common.cuh"

// Mirai S linears (ggml_mirai_quantize / ggml_mirai_mul_mat), ported from Mirai's vLLM plugin kernels.
bool ggml_cuda_mirai_supports_op(const ggml_tensor * op);

void ggml_cuda_op_mirai_quantize(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_mirai_mul_mat(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
