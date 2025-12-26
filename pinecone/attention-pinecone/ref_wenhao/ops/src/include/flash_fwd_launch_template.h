/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once
#include "namespace_config.h"
#include <c10/cuda/CUDAException.h>  // For C10_CUDA_CHECK and C10_CUDA_KERNEL_LAUNCH_CHECK

#include "static_switch.h"
#include "hardware_info.h"
#include "flash.h"
#include "flash_fwd_kernel.h"
#include "bind_fwd_kernel.h"

namespace FLASH_NAMESPACE {

// Determine if the architecture supports FLASH and define a macro to handle parameter modifiers
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define ARCH_SUPPORTS_FLASH
#define KERNEL_PARAM_MODIFIER __grid_constant__
#else
#define KERNEL_PARAM_MODIFIER
#endif

// Define a macro for unsupported architecture handling to centralize the error message
#define FLASH_UNSUPPORTED_ARCH printf("FATAL: FlashAttention requires building with sm version sm80-sm90, but was built for < 8.0!");

// Use a macro to clean up kernel definitions
#define DEFINE_FLASH_FORWARD_KERNEL(kernelName, ...) \
template<typename Kernel_traits, __VA_ARGS__> \
__global__ void kernelName(KERNEL_PARAM_MODIFIER const Flash_fwd_params params)

DEFINE_FLASH_FORWARD_KERNEL(flash_fwd_kernel, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax) {
    #if defined(ARCH_SUPPORTS_FLASH)
        static_assert(!(Is_causal && Is_local)); // Enforce constraints
        FLASH_NAMESPACE::compute_attn<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Return_softmax>(params);
    #else
        FLASH_UNSUPPORTED_ARCH
    #endif
}

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax>
__global__ void bind_fwd_kernel(const Flash_fwd_params params,
    const int* full_row_ptr, const int* full_col_idx,
    const int* part_row_ptr, const int* part_col_idx, uint64_t* inner_bitmaps,
    const int* load_row_ptr, const int* load_col_idx) {

    static_assert(!(Is_causal && Is_local)); // Enforce constraints
    FLASH_NAMESPACE::compute_mask_attn<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Return_softmax>(params,
    full_row_ptr, full_col_idx, 
    part_row_ptr, part_col_idx, inner_bitmaps,
    load_row_ptr, load_col_idx);
}

template<typename Kernel_traits, bool Is_dropout, bool Is_causal>
void run_flash_fwd(Flash_fwd_params &params, cudaStream_t stream,
    const int* full_row_ptr, const int* full_col_idx,
    const int* part_row_ptr, const int* part_col_idx, uint64_t* inner_bitmaps,
    const int* load_row_ptr, const int* load_col_idx) {
    constexpr size_t smem_size = Kernel_traits::kSmemSize;
    // printf("smem_size = %d\n", smem_size);

    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    dim3 grid(num_m_block, params.b, params.h);
    const bool is_even_MN = params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr && params.seqlen_k % Kernel_traits::kBlockN == 0 && params.seqlen_q % Kernel_traits::kBlockM == 0;
    const bool is_even_K = params.d == Kernel_traits::kHeadDim;
    const bool return_softmax = params.p_ptr != nullptr;
    BOOL_SWITCH(is_even_MN, IsEvenMNConst, [&] {
        EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
            LOCAL_SWITCH((params.window_size_left >= 0 || params.window_size_right >= 0) && !Is_causal, Is_local, [&] {
                BOOL_SWITCH(return_softmax, ReturnSoftmaxConst, [&] {
                    ALIBI_SWITCH(params.alibi_slopes_ptr != nullptr, Has_alibi, [&] {
                        SOFTCAP_SWITCH(params.softcap > 0.0, Is_softcap, [&] {

                            // auto kernel = &flash_fwd_kernel<Kernel_traits, Is_dropout && !Is_softcap, Is_causal, Is_local && !Is_causal, Has_alibi, IsEvenMNConst && IsEvenKConst && !Is_local && !Has_alibi && !ReturnSoftmaxConst && Kernel_traits::kHeadDim <= 128, IsEvenKConst && !ReturnSoftmaxConst && !Has_alibi, Is_softcap, ReturnSoftmaxConst && Is_dropout && !Is_softcap>;
                            
                            auto kernel = &bind_fwd_kernel<Kernel_traits, Is_dropout && !Is_softcap, Is_causal, Is_local && !Is_causal, Has_alibi, IsEvenMNConst && IsEvenKConst && !Is_local && !Has_alibi && !ReturnSoftmaxConst && Kernel_traits::kHeadDim <= 128, IsEvenKConst && !ReturnSoftmaxConst && !Has_alibi, Is_softcap, ReturnSoftmaxConst && Is_dropout && !Is_softcap>;

                            // printf("IsEvenMNConst = %d, IsEvenKConst = %d, Is_local = %d, Is_causal = %d, ReturnSoftmaxConst = %d, Is_dropout = %d\n", int(IsEvenMNConst), int(IsEvenKConst), int(Is_local), int(Is_causal), int(ReturnSoftmaxConst), int(Is_dropout));

                            if (smem_size >= 48 * 1024) {
                                C10_CUDA_CHECK(cudaFuncSetAttribute(
                                    kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
                            }
                            // int ctas_per_sm;
                            // cudaError status_ = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                            //     &ctas_per_sm, kernel, Kernel_traits::kNThreads, smem_size);
                            // printf("smem_size = %d, CTAs per SM = %d\n", int(smem_size), ctas_per_sm);
                            
                            // 正经的启动kernel
                            kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(
                                params, 
                                full_row_ptr, full_col_idx, 
                                part_row_ptr, part_col_idx, inner_bitmaps,
                                load_row_ptr, load_col_idx);
                            C10_CUDA_KERNEL_LAUNCH_CHECK();
                        });
                    });
                });
            });
        });
    });
}


// dwh：.cu文件调用它时 run_mha_fwd_hdim64<cutlass::half_t, false>(params, stream);
template<typename T, bool Is_causal>
void run_mha_fwd_hdim64(Flash_fwd_params &params, cudaStream_t stream, 
    const int* full_row_ptr, const int* full_col_idx,
    const int* part_row_ptr, const int* part_col_idx, uint64_t* inner_bitmaps,
    const int* load_row_ptr, const int* load_col_idx) {
    constexpr static int Headdim = 64;

    constexpr static bool Is_dropout = false; // 保持
    // Using 8 warps is 18% slower for seqlen=2k, 2 warps is 5% slower
    // Using block size (64 x 256) is 27% slower for seqlen=2k
    // Using block size (256 x 64) is 85% slower for seqlen=2k, because of register spilling

    // dwh: 对应关系为 <kHeadDim_, kBlockM_, kBlockN_, kNWarps_, Is_Q_in_regs_=false, bool Share_Q_K_smem_=false, typename elem_type=cutlass::half_t>. 
    // dwh: 其中 Is_Q_in_regs_ 决定是否让 Q矩阵保留在寄存器减少共享内存访问
    // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 128, 4, false, false, T>, Is_dropout, Is_causal>(
    //     params, stream, 
    //     full_row_ptr, full_col_idx, 
    //     part_row_ptr, part_col_idx, inner_bitmaps,
    //     load_row_ptr, load_col_idx);
    // dwh: 注意这个参数设置 需要对应到  kernel_traits.h 的 Line51 来看
    // dwh: 得知了 其 blocksize 的设置几乎总是 4 warp = 128


    // 同时启用  Is_Q_in_regs_ 和 Share_Q_K_smem_
    // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 128, 4, true, true, T>, Is_dropout, Is_causal>(
    //     params, stream, 
    //     full_row_ptr, full_col_idx, 
    //     part_row_ptr, part_col_idx, inner_bitmaps,
    //     load_row_ptr, load_col_idx);

    // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, true, true, T>, Is_dropout, Is_causal>(
    //     params, stream, 
    //     full_row_ptr, full_col_idx, 
    //     part_row_ptr, part_col_idx, inner_bitmaps,
    //     load_row_ptr, load_col_idx);

    // 性能测试时 最好的配置
    run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, true, true, T>, Is_dropout, Is_causal>(
        params, stream, 
        full_row_ptr, full_col_idx, 
        part_row_ptr, part_col_idx, inner_bitmaps,
        load_row_ptr, load_col_idx);

    
    // // 只开启用  Is_Q_in_regs_ 
    // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 128, 4, true, false, T>, Is_dropout, Is_causal>(
    //     params, stream, 
    //     full_row_ptr, full_col_idx, 
    //     part_row_ptr, part_col_idx, inner_bitmaps,
    //     load_row_ptr, load_col_idx);


    
    // // 只开启  Share_Q_K_smem_
    // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 128, 4, false, true, T>, Is_dropout, Is_causal>(
    //     params, stream, 
    //     full_row_ptr, full_col_idx, 
    //     part_row_ptr, part_col_idx, inner_bitmaps,
    //     load_row_ptr, load_col_idx);

    // 原始的推荐 开启符号 / block 尺寸信息
    // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 128, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
    // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, true, false, T>, Is_dropout, Is_causal>(params, stream);
    // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, true, true, T>, Is_dropout, Is_causal>(params, stream);
    
}

}
