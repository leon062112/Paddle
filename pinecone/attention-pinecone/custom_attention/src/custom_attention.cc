// pytorch相关头文件
// #include <ATen/ATen.h>
// #include <torch/extension.h>
// #include <torch/python.h>
// #include <torch/nn/functional.h>
// #include <c10/cuda/CUDAGuard.h>
// #include <c10/cuda/CUDAStream.h>
// #include <ATen/cuda/CUDAGeneratorImpl.h>  // For at::Generator and at::PhiloxCudaState
// #include "include/philox_unpack.cuh"  // For at::cuda::philox::unpack

#include <cuda_fp16.h> 
#include <cutlass/numeric_types.h>
#include "include/namespace_config.h"
#include "include/hardware_info.h"
#include "include/flash.h"
#include "include/static_switch.h"
#include "paddle/extension.h"
#include "paddle/phi/api/include/api.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/ddim.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/platform/device_context.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include <cuda_runtime.h>


#define CHECK_DEVICE(x) PD_CHECK((x).is_gpu(), #x "must be on CUDA")
#define CHECK_SHAPE(x, ...) PD_CHECK((x).dims() == phi::make_ddim({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")
// #define CHECK_CONTIGUOUS(x) PD_CHECK((x).is_contiguous(), #x " must be contiguous")

namespace FLASH_NAMESPACE {

//stride怎么用   
//done
void set_params_fprop(Flash_fwd_params &params,
                      // sizes
                      const size_t b,
                      const size_t seqlen_q,
                      const size_t seqlen_k,
                      const size_t seqlen_q_rounded,
                      const size_t seqlen_k_rounded,
                      const size_t h,
                      const size_t h_k,
                      const size_t d,
                      const size_t d_rounded,
                    //   device pointers
                    //   const at::Tensor q,
                    //   const at::Tensor k,
                    //   const at::Tensor v,
                    //   at::Tensor out,
                      const paddle::Tensor& q,
                      const paddle::Tensor& k,
                      const paddle::Tensor& v,
                      paddle::Tensor& out,
                      void *cu_seqlens_q_d,
                      void *cu_seqlens_k_d,
                      void *seqused_k,
                      void *p_d,
                      void *softmax_lse_d,
                      float p_dropout,
                      float softmax_scale,
                      int window_size_left,
                      int window_size_right,
                      const float softcap,
                      bool seqlenq_ngroups_swapped=false,
                      const bool unpadded_lse=false) {

    // Reset the parameters
    params = {};
    // params.is_bf16 = q.dtype() == torch::kBFloat16;
    params.is_bf16 = (q.dtype() == phi::DataType::BFLOAT16);
    // Set the pointers and strides.
    params.q_ptr = const_cast<void*>(q.data());
    params.k_ptr = const_cast<void*>(k.data());
    params.v_ptr = const_cast<void*>(v.data());
    // All stride are in elements, not bytes.
    PD_CHECK(q.strides().size() >= 3, "q must be at least 3D");
    PD_CHECK(k.strides().size() >= 3, "k must be at least 3D");
    PD_CHECK(v.strides().size() >= 3, "v must be at least 3D");
    PD_CHECK(out.strides().size() >= 3, "out must be at least 3D");

    params.q_row_stride = q.strides()[q.strides().size() - 3];
    params.k_row_stride = k.strides()[k.strides().size() - 3];
    params.v_row_stride = v.strides()[v.strides().size() - 3];

    params.q_head_stride = q.strides()[q.strides().size() - 2];
    params.k_head_stride = k.strides()[k.strides().size() - 2];
    params.v_head_stride = v.strides()[v.strides().size() - 2];

    params.o_ptr = out.data();
    params.o_row_stride = out.strides()[out.strides().size() - 3];
    params.o_head_stride = out.strides()[out.strides().size() - 2];

    if (cu_seqlens_q_d == nullptr) {
    params.q_batch_stride = q.strides()[0];
    params.k_batch_stride = k.strides()[0];
    params.v_batch_stride = v.strides()[0];
    params.o_batch_stride = out.strides()[0];
        if (seqlenq_ngroups_swapped) {
             params.q_batch_stride *= seqlen_q;
             params.o_batch_stride *= seqlen_q;
        }
    }

    params.cu_seqlens_q = static_cast<int *>(cu_seqlens_q_d);
    params.cu_seqlens_k = static_cast<int *>(cu_seqlens_k_d);
    params.seqused_k = static_cast<int *>(seqused_k);

    // P = softmax(QK^T)
    params.p_ptr = p_d;

    // Softmax sum
    params.softmax_lse_ptr = softmax_lse_d;

    // Set the dimensions.
    params.b = b;
    params.h = h;
    params.h_k = h_k;
    params.h_h_k_ratio = h / h_k;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.seqlen_q_rounded = seqlen_q_rounded;
    params.seqlen_k_rounded = seqlen_k_rounded;
    params.d = d;
    params.d_rounded = d_rounded;

    // Set the different scale values.
    #ifdef FLASHATTENTION_DISABLE_SOFTCAP
        PD_CHECK(softcap <= 0.0, "This flash attention build does not support softcap.");
    #endif
    if (softcap > 0.0) {
        params.softcap = softmax_scale / softcap;
        params.scale_softmax = softcap;
        params.scale_softmax_log2 = softcap * M_LOG2E;
    } else{
        // Remove potential NaN
        params.softcap = 0.0;
        params.scale_softmax = softmax_scale;
        params.scale_softmax_log2 = softmax_scale * M_LOG2E;
    }

    // Set this to probability of keeping an element to simplify things.
    params.p_dropout = 1.f - p_dropout;
    // Convert p from float to int so we don't have to convert the random uint to float to compare.
    // [Minor] We want to round down since when we do the comparison we use <= instead of <
    // params.p_dropout_in_uint = uint32_t(std::floor(params.p_dropout * 4294967295.0));
    // params.p_dropout_in_uint16_t = uint16_t(std::floor(params.p_dropout * 65535.0));
    params.p_dropout_in_uint8_t = uint8_t(std::floor(params.p_dropout * 255.0));
    params.rp_dropout = 1.f / params.p_dropout;
    params.scale_softmax_rp_dropout = params.rp_dropout * params.scale_softmax;
    PD_CHECK(p_dropout < 1.f, "p_dropout must be less than 1.0");
    #ifdef FLASHATTENTION_DISABLE_DROPOUT
        PD_CHECK(p_dropout == 0.0f, "This flash attention build does not support dropout.");
    #endif

    // Causal is the special case where window_size_right == 0 and window_size_left < 0.
    // Local is the more general case where window_size_right >= 0 or window_size_left >= 0.
    params.is_causal = window_size_left < 0 && window_size_right == 0;

    if (window_size_left < 0 && window_size_right >= 0) { window_size_left = seqlen_k; }
    if (window_size_left >= 0 && window_size_right < 0) { window_size_right = seqlen_k; }
    params.window_size_left = window_size_left;
    params.window_size_right = window_size_right;

    #ifdef FLASHATTENTION_DISABLE_LOCAL
        PD_CHECK(params.is_causal || (window_size_left < 0 && window_size_right < 0),
            "This flash attention build does not support local attention.");
    #endif

    params.is_seqlens_k_cumulative = true;

    #ifdef FLASHATTENTION_DISABLE_UNEVEN_K
        PD_CHECK(d == d_rounded, "This flash attention build does not support headdim not being a multiple of 32.");
    #endif

    params.unpadded_lse = unpadded_lse;
    params.seqlenq_ngroups_swapped = seqlenq_ngroups_swapped;
}

//要保留吗
// void set_params_alibi(Flash_fwd_params &params, std::optional<at::Tensor> &alibi_slopes_, int batch_size, int num_heads){
// #ifdef FLASHATTENTION_DISABLE_ALIBI
//     TORCH_CHECK(!alibi_slopes_.has_value(), "This flash attention build does not support alibi.");
//     params.alibi_slopes_ptr = nullptr;
// #else
//     if (alibi_slopes_.has_value()) {
//         auto alibi_slopes = alibi_slopes_.value();
//         TORCH_CHECK(alibi_slopes.dtype() == torch::kFloat32, "ALiBi slopes must have dtype fp32");
//         CHECK_DEVICE(alibi_slopes);
//         TORCH_CHECK(alibi_slopes.stride(-1) == 1, "ALiBi slopes tensor must have contiguous last dimension");
//         TORCH_CHECK(alibi_slopes.sizes() == torch::IntArrayRef({num_heads}) || alibi_slopes.sizes() == torch::IntArrayRef({batch_size, num_heads}));
//         params.alibi_slopes_ptr = alibi_slopes.data_ptr();
//         params.alibi_slopes_batch_stride = alibi_slopes.dim() == 2 ? alibi_slopes.stride(0) : 0;
//     } else {
//         params.alibi_slopes_ptr = nullptr;
//     }
// #endif
// }


//不用改
inline int num_splits_heuristic(int batch_nheads_mblocks, int num_SMs, int num_n_blocks, int max_splits) {
    // If we have enough to almost fill the SMs, then just use 1 split
    if (batch_nheads_mblocks >= 0.8f * num_SMs) { return 1; }
    max_splits = std::min({max_splits, num_SMs, num_n_blocks});
    float max_efficiency = 0.f;
    std::vector<float> efficiency;
    efficiency.reserve(max_splits);
    auto ceildiv = [](int a, int b) { return (a + b - 1) / b; };
    // Some splits are not eligible. For example, if we have 64 blocks and choose 11 splits,
    // we'll have 6 * 10 + 4 blocks. If we choose 12 splits, we'll have 6 * 11 + (-2) blocks
    // (i.e. it's 11 splits anyway).
    // So we check if the number of blocks per split is the same as the previous num_splits.
    auto is_split_eligible = [&ceildiv, &num_n_blocks](int num_splits) {
        return num_splits == 1 || ceildiv(num_n_blocks, num_splits) != ceildiv(num_n_blocks, num_splits - 1);
    };
    for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
        if (!is_split_eligible(num_splits)) {
            efficiency.push_back(0.f);
        } else {
            float n_waves = float(batch_nheads_mblocks * num_splits) / num_SMs;
            float eff = n_waves / ceil(n_waves);
            // printf("num_splits = %d, eff = %f\n", num_splits, eff);
            if (eff > max_efficiency) { max_efficiency = eff; }
            efficiency.push_back(eff);
        }
    }
    for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
        if (!is_split_eligible(num_splits)) { continue; }
        if (efficiency[num_splits - 1] >= 0.85 * max_efficiency) {
            // printf("num_splits chosen = %d\n", num_splits);
            return num_splits;
        }
    }
    return 1;
}

std::tuple<paddle::Tensor, paddle::Tensor> set_params_splitkv(
    Flash_fwd_params &params,
    const int batch_size,
    const int num_heads,
    const int head_size,
    const int max_seqlen_k,
    const int max_seqlen_q,
    const int head_size_rounded,
    const float p_dropout,
    const int num_splits,
    const int num_sm,
    const paddle::Place& place = paddle::GPUPlace()  // 仅保留 place，默认 GPU
) {
    const int block_n = head_size <= 64 ? 256 : (head_size <= 128 ? 128 : 64);
    const int num_n_blocks = (max_seqlen_k + block_n - 1) / block_n;
    const int num_m_blocks = (max_seqlen_q + 64 - 1) / 64;

    params.num_splits = num_splits;
    paddle::Tensor softmax_lse_accum;
    paddle::Tensor out_accum;

    if (p_dropout == 0.0f) {
        if (num_splits < 1) {
            params.num_splits = num_splits_heuristic(
                batch_size * num_heads * num_m_blocks, num_sm * 2, num_n_blocks, 128);
        }
        if (params.num_splits > 1) {
            softmax_lse_accum = paddle::empty(
                {params.num_splits, batch_size, num_heads, max_seqlen_q},
                paddle::DataType::FLOAT32,
                place
            );
            out_accum = paddle::empty(
                {params.num_splits, batch_size, num_heads, max_seqlen_q, head_size_rounded},
                paddle::DataType::FLOAT32,
                place
            );
            params.softmax_lseaccum_ptr = const_cast<void*>(softmax_lse_accum.data());
            params.oaccum_ptr = const_cast<void*>(out_accum.data());
        }
        PD_CHECK(params.num_splits <= 128, "num_splits > 128 not supported");
    }

    return std::make_tuple(softmax_lse_accum, out_accum);
}


void run_mha_fwd(
        Flash_fwd_params &params, 
        cudaStream_t stream, 
        const int* full_row_ptr, 
        const int* full_col_idx,
        const int* part_row_ptr, 
        const int* part_col_idx, 
        const uint64_t* inner_bitmaps,
        const int* load_row_ptr, 
        const int* load_col_idx
    ){
    // 支持 cutlass::half_t, HeadDim=64, causal=true
    if (params.d != 64) {
        PD_THROW("Only head_dim=64 is supported");
    }
    if (params.is_bf16) {
        PD_THROW("Only fp16 is supported, not bf16");
    }

    if (params.is_causal) {
        run_mha_fwd_<cutlass::half_t, 64, true>(params, stream, 
            full_row_ptr, full_col_idx, 
            part_row_ptr, part_col_idx, inner_bitmaps,
            load_row_ptr, load_col_idx
        );
    } else {
        run_mha_fwd_<cutlass::half_t, 64, false>(params, stream, 
            full_row_ptr, full_col_idx, 
            part_row_ptr, part_col_idx, inner_bitmaps,
            load_row_ptr, load_col_idx
        );
    }
}

std::vector<paddle::Tensor> flashattn_binding_gpu(
        paddle::Tensor &q_in,         // batch_size x seqlen_q x num_heads x round_multiple(head_size, 8)
        const paddle::Tensor &k,         // batch_size x seqlen_k x num_heads_k x round_multiple(head_size, 8)
        const paddle::Tensor &v,         // batch_size x seqlen_k x num_heads_k x round_multiple(head_size, 8)
        const paddle::Tensor &full_row_ptr, 
        const paddle::Tensor &full_col_idx,
        const paddle::Tensor &part_row_ptr, 
        const paddle::Tensor &part_col_idx, 
        const paddle::Tensor &inner_bitmaps,
        const paddle::Tensor &load_row_ptr, 
        const paddle::Tensor &load_col_idx,
        // std::optional<paddle::Tensor> &out_,          // batch_size x seqlen_q x num_heads x round_multiple(head_size, 8)
        // std::optional<paddle::Tensor> &alibi_slopes_, // num_heads or batch_size x num_heads
        const float p_dropout,
        const float softmax_scale,
        bool is_causal,
        int window_size_left,
        int window_size_right,
        const float softcap,
        const bool return_softmax
        // std::optional<at::Generator> gen_
    ) {

    auto place = q_in.place();
    auto* dev_ctx = phi::DeviceContextPool::Instance().Get(place);
    auto* gpu_ctx = dynamic_cast<const phi::GPUContext*>(dev_ctx);
    PD_CHECK(gpu_ctx != nullptr, "Failed to get phi::GPUContext from DeviceContextPool");

    cudaStream_t stream = gpu_ctx->stream();

    auto [cc_major, cc_minor] = get_compute_capability(get_current_device());
    bool is_sm8x_min = cc_major >= 8;
    PD_CHECK(is_sm8x_min, "FlashAttention only supports Ampere GPUs or newer.");
    paddle::Tensor q = q_in;
    auto q_dtype = q.dtype();
    PD_CHECK(q_dtype == paddle::DataType::FLOAT16 || q_dtype == paddle::DataType::BFLOAT16,
                "FlashAttention only support fp16 and bf16 data type");
    PD_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    PD_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");

    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);

    // TODO
    // PD_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    // PD_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    // PD_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");

    // const auto sizes = q.sizes();
    const auto q_shape = q.shape(); 

    const int batch_size = q_shape[0];
    int seqlen_q = q_shape[1];
    int num_heads = q_shape[2];
    const int head_size = q_shape[3];
    // const int seqlen_k = k.size(1);
    // const int num_heads_k = k.size(2);

    const auto k_shape = k.shape(); 
    const int seqlen_k = k_shape[1];
    const int num_heads_k = k_shape[2];

    PD_CHECK(batch_size > 0, "batch size must be positive");
    PD_CHECK(head_size <= 256, "FlashAttention forward only supports head dimension at most 256");
    PD_CHECK(head_size % 8 == 0, "query, key, value, and out_ must have a head_size that is a multiple of 8");
    PD_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");


    // std::cout << ">>> [DAVID INFO] batch_size="<<batch_size<<"; head_num:="<<num_heads
    //     <<"; seq_len="<<seqlen_q<<"; head_size="<<head_size<<std::endl;


    if (softcap > 0.f) { PD_CHECK(p_dropout == 0.f, "Softcapping does not support dropout for now"); }

    if (window_size_left >= seqlen_k) { window_size_left = -1; }
    if (window_size_right >= seqlen_k) { window_size_right = -1; }

    // causal=true is the same as causal=false in this case
    // if (seqlen_q == 1 && !alibi_slopes_.has_value()) { is_causal = false; }
    if (seqlen_q == 1) { is_causal = false; }
    if (is_causal) { window_size_right = 0; }

    // Faster to transpose q from (b, 1, (nheads_kv ngroups), d) to (b, ngroups, nheads_kv, d) in this case
    // H/t Daniel Haziza
    // const int seqlenq_ngroups_swapped = 
    //     seqlen_q == 1 && num_heads > num_heads_k && 
    //     window_size_left < 0 && window_size_right < 0 && 
    //     p_dropout == 0.f && head_size % 8 == 0 && !alibi_slopes_.has_value();
    const int seqlenq_ngroups_swapped =
        seqlen_q == 1 && num_heads > num_heads_k &&
        window_size_left < 0 && window_size_right < 0 &&
        p_dropout == 0.f && head_size % 8 == 0;
        const int ngroups = num_heads / num_heads_k;
    if (seqlenq_ngroups_swapped) {
        q = paddle::experimental::reshape(q, {batch_size, num_heads_k, ngroups, head_size});
        q = paddle::experimental::transpose(q, std::vector<int>{0, 2, 1, 3});
        seqlen_q = ngroups;
        num_heads = num_heads_k;
    }

    CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size);
    CHECK_SHAPE(k, batch_size, seqlen_k, num_heads_k, head_size);
    CHECK_SHAPE(v, batch_size, seqlen_k, num_heads_k, head_size);

    paddle::Tensor out = paddle::empty_like(q);
    // if (out_.has_value()) {
    //     out = out_.value();
    //     PD_CHECK(out.dtype() == q_dtype, "Output must have the same dtype as inputs");
    //     CHECK_DEVICE(out);
    //     // PD_CHECK(out.stride(-1) == 1, "Output tensor must have contiguous last dimension");
    //     // CHECK_SHAPE(out, batch_size, sizes[1], sizes[2], head_size);
    //     if (seqlenq_ngroups_swapped) {
    //         out = paddle::experimental::reshape(out, {batch_size, num_heads_k, ngroups, head_size});
    //         out = paddle::experimental::transpose(out, std::vector<int>{0, 2, 1, 3});
    //     }
    // } else {
    //     out = paddle::empty_like(q);
    // }

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size_rounded = round_multiple(head_size, head_size <= 128 ? 32 : 64);
    const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
    const int seqlen_k_rounded = round_multiple(seqlen_k, 128);

    // auto opts = q.options();
    auto dtype = q.dtype();          // paddle::DataType
    // auto place = q.place();          // paddle::Place (GPUPlace)

    // auto softmax_lse = paddle::empty({batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));
    auto softmax_lse = paddle::empty({batch_size, num_heads, seqlen_q}, paddle::DataType::FLOAT32,place);

    paddle::Tensor p;
    // Only return softmax if there's dropout to reduce compilation time
    if (return_softmax) {
        PD_CHECK(p_dropout > 0.0f, "return_softmax is only supported when p_dropout > 0.0");
        p = paddle::empty({ batch_size, num_heads, seqlen_q_rounded, seqlen_k_rounded }, dtype,place);
    }
    else {
        p = paddle::empty({ 0 },dtype,place);
    }

    Flash_fwd_params params = {};
    set_params_fprop(params,
                     batch_size,
                     seqlen_q, seqlen_k,
                     seqlen_q_rounded, seqlen_k_rounded,
                     num_heads, num_heads_k,
                     head_size, head_size_rounded,
                     q, k, v, out,
                     /*cu_seqlens_q_d=*/nullptr,
                     /*cu_seqlens_k_d=*/nullptr,
                     /*seqused_k=*/nullptr,
                     return_softmax ? p.data() : nullptr,
                     softmax_lse.data(),
                     p_dropout,
                     softmax_scale,
                     window_size_left,
                     window_size_right,
                     softcap
                     );

    // Keep references to these tensors to extend their lifetime
    paddle::Tensor softmax_lse_accum, out_accum;
    std::tie(softmax_lse_accum, out_accum) = set_params_splitkv(
        params, batch_size, num_heads, head_size, seqlen_k, seqlen_q,
        head_size_rounded, p_dropout, /*num_splits*/ 0, get_num_sm(get_current_device()),place);

    // number of times random will be generated per thread, to offset philox counter in thc random
    // state
    // We use a custom RNG that increases the offset by batch_size * nheads * 32.
    int64_t counter_offset = params.b * params.h * 32;
    // auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    // auto rng_state = paddle::empty({2}, options.dtype(torch::kInt64));
    // Forward kernel will populate memory with the seed and offset.
    // params.rng_state = reinterpret_cast<uint64_t*>(rng_state.data_ptr());

    PD_CHECK(p_dropout == 0.0f, "This Paddle FlashAttention build currently does not support dropout (p_dropout must be 0).");
    PD_CHECK(!return_softmax, "return_softmax requires dropout; currently unsupported.");
    // auto place = q.place(); // 假设 q 是输入 tensor，已知在 GPU
    auto rng_state = paddle::empty({2}, paddle::DataType::INT64, place);

    // 获取原始指针（Paddle 使用 .data_ptr()）
    // params.rng_state = reinterpret_cast<uint64_t*>(rng_state.data<int64_t>());

    // if (p_dropout > 0.0)  {
    //     auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
    //         gen_, at::cuda::detail::getDefaultCUDAGenerator());
    //     // See Note [Acquire lock when using random generators]
    //     std::lock_guard<std::mutex> lock(gen->mutex_);
    //     params.philox_args = gen->philox_cuda_state(counter_offset);
    // }

    // set_params_alibi(params, alibi_slopes_, batch_size, num_heads);

    if (seqlen_k > 0) {
        run_mha_fwd(params, stream, 
            reinterpret_cast<const int*>(full_row_ptr.data()),
            reinterpret_cast<const int*>(full_col_idx.data()),
            reinterpret_cast<const int*>(part_row_ptr.data()),
            reinterpret_cast<const int*>(part_col_idx.data()),
            reinterpret_cast<const uint64_t*>(inner_bitmaps.data()),
            reinterpret_cast<const int*>(load_row_ptr.data()),
            reinterpret_cast<const int*>(load_col_idx.data())
            // full_row_ptr.data_ptr<int>(),
            // full_col_idx.data_ptr<int>(),
            // part_row_ptr.data_ptr<int>(),
            // part_col_idx.data_ptr<int>(),
            // // reinterpret_cast< __half*>(part_block_mask.data_ptr<at::Half>()),
            // // reinterpret_cast<uint64_t*>(inner_bitmaps.data_ptr<int64_t>()),
            // reinterpret_cast<uint64_t*>(inner_bitmaps.data()),
            // load_row_ptr.data_ptr<int>(),
            // load_col_idx.data_ptr<int>()
        );
    } else {
        // If seqlen_k == 0, then we have an empty tensor. We need to set the output to 0.
        // out.zero_();
        // softmax_lse.fill_(std::numeric_limits<float>::infinity());
        out = paddle::full(out.shape(), 0.0f, out.dtype(), out.place());
        softmax_lse = paddle::full(softmax_lse.shape(),std::numeric_limits<float>::infinity(),softmax_lse.dtype(),softmax_lse.place());
    }

    if (seqlenq_ngroups_swapped) {
        // out = out.transpose(1, 2).reshape({batch_size, 1, num_heads_k * seqlen_q, head_size});
        // q = q.transpose(1, 2).reshape({batch_size, 1, num_heads_k * seqlen_q, head_size});
        // softmax_lse = softmax_lse.reshape({batch_size, num_heads_k * seqlen_q, 1});
        out = paddle::experimental::reshape(out, {batch_size, num_heads_k, ngroups, head_size});
        out = paddle::experimental::transpose(out, std::vector<int>{0, 2, 1, 3});
        q = paddle::experimental::reshape(q, {batch_size, num_heads_k, ngroups, head_size});
        q = paddle::experimental::transpose(q, std::vector<int>{0, 2, 1, 3});
        softmax_lse = paddle::experimental::reshape(softmax_lse, {batch_size, num_heads_k * seqlen_q, 1});

    }
    return {out, softmax_lse, p, rng_state};
}



} // namespace FLASH_NAMESPACE


// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
// {
//     m.doc() = "Flash_Attn_Binding: Test for Pinecone Fund";
//     m.def("forward", &FLASH_NAMESPACE::flashattn_binding_gpu, "FlashAttn Binded op Forward"); 
// }

PD_BUILD_OP(custom_attention)
    .Inputs({"q", "k", "v", 
             "full_row_ptr", "full_col_idx",
             "part_row_ptr", "part_col_idx", 
             "inner_bitmaps",
             "load_row_ptr", "load_col_idx"})
    .Outputs({"out", "softmax_lse", "p", "rng_state"})
    .Attrs({"p_dropout: float",
            "softmax_scale: float",
            "is_causal: bool",
            "window_size_left: int",
            "window_size_right: int",
            "softcap: float",
            "return_softmax: bool"})
    .SetKernelFn(PD_KERNEL(FLASH_NAMESPACE::flashattn_binding_gpu));