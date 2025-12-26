//  /flashmask-test/flash-attention/csrc/flash_attn/src/flash_fwd_hdim64_fp16_causal_sm80.cu

// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"


#include "include/namespace_config.h"
#include "include/flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {


// flash_fwd_hdim64_fp16_sm80
template<>
void run_mha_fwd_<cutlass::half_t, 64, false>(Flash_fwd_params &params, cudaStream_t stream, 
    const int* full_row_ptr, const int* full_col_idx,
    const int* part_row_ptr, const int* part_col_idx, uint64_t* inner_bitmaps,
    const int* load_row_ptr, const int* load_col_idx) {
    run_mha_fwd_hdim64<cutlass::half_t, false>(params, stream, 
        full_row_ptr, full_col_idx, 
        part_row_ptr, part_col_idx, inner_bitmaps,
        load_row_ptr, load_col_idx);
}


// flash_fwd_hdim64_fp16_causal_sm80
template<>
void run_mha_fwd_<cutlass::half_t, 64, true>(Flash_fwd_params &params, cudaStream_t stream,
    const int* full_row_ptr, const int* full_col_idx,
    const int* part_row_ptr, const int* part_col_idx, uint64_t* inner_bitmaps,
    const int* load_row_ptr, const int* load_col_idx) {
    run_mha_fwd_hdim64<cutlass::half_t, true>(params, stream,
        full_row_ptr, full_col_idx, 
        part_row_ptr, part_col_idx, inner_bitmaps,
        load_row_ptr, load_col_idx);
}

} // namespace FLASH_NAMESPACE