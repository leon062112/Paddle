# 2025.10.30 Thur.
# 
# 基于 https://github.com/HeyDavid633/flashmask-test/blob/main/binding/correct_verify4.py
# 将代码改造为 paddle 的形式，并且进行初步的 cuda 算子绑定过程
#
# 在 bitmap_mask.py 以后真实地传入给 cuda 这个 unit64的数组
# 需要在这里验证 自己的 mask 加的是否正确，由于分块尺寸 64 * 64；
# 所以 bitmap 的存储应该是 64 个 unint_64 的数据
# 
# python correct_verify.py --batch_size 1 --head_num 1 --head_size 64 --seq_len 128
import sys
import os
import argparse
import paddle
import math
# import matplotlib.pyplot as plt
import numpy as np
import time
from util.masks import generate_causal_mask, generate_full_mask
from util.utils import set_dtype, seqlen_to_mask, paddle_device_identify
from custom_attention import custom_attention

def attention_paddle(q, k, v, dropout_p=0.0, causal=True):
    batch_size, seqlen, nheads, d = q.shape
    q = q.reshape([batch_size * nheads, seqlen, d])
    k = k.reshape([batch_size * nheads, d, seqlen])
    
    softmax_scale = 1.0 / math.sqrt(d)
    scores = paddle.bmm(q, k) * softmax_scale
    
    if causal:
        causal_mask = paddle.triu(paddle.full([seqlen, seqlen], -10000.0), 1)
        scores = scores + causal_mask.astype(scores.dtype)
    
    attention = paddle.nn.functional.softmax(scores, axis=-1)
    attention_drop = paddle.nn.functional.dropout(attention, p=dropout_p)
    output = paddle.bmm(attention_drop, v.reshape([batch_size * nheads, seqlen, d]))
    return output.reshape([batch_size, seqlen, nheads, d])

def block_to_bitmap(block):
    """Convert 8x8 block to uint64 bitmap"""
    assert block.shape == (8, 8), "Block size must be 8x8"
    bitmap = np.uint64(0)
    for i in range(8):
        for j in range(8):
            if block[i, j] != 0:
                bitmap |= np.uint64(1) << np.uint64(i * 8 + j)
    return bitmap

def bitmap_to_matrix(bitmap):
    """Convert bitmap to matrix"""
    matrix = paddle.zeros([8, 8], dtype='uint8')
    
    if isinstance(bitmap, paddle.Tensor):
        bitmap = int(bitmap.item())
    bitmap = int(bitmap)
    
    if bitmap == 0:
        return matrix
    
    for pos in range(64):
        if bitmap & (1 << pos):
            i, j = divmod(pos, 8)
            matrix[i, j] = 1
    return matrix

def get_InnerTile_bitmap(outer_tile):
    """Convert OuterTile to InnerTile bitmaps"""
    bitmaps = []
    outer_tile_size = outer_tile.shape[0]
    
    for j in range(0, outer_tile_size, 8):
        for i in range(0, outer_tile_size, 8):
            inner_tile = outer_tile[i:i+8, j:j+8]
            bitmap = 0
            for bi in range(8):
                for bj in range(8):
                    if inner_tile[bi, bj] != 0:
                        bitmap |= 1 << (bi * 8 + bj)
            bitmaps.append(bitmap)
    return bitmaps

def get_OuterTile_storage(Mask, block_size_m=32, block_size_n=32):
    """Generate sparse storage structure for OuterTiles"""
    batch_size, n, _ = Mask.shape
    total_elements = n * n
    nnz = paddle.count_nonzero(Mask) / total_elements * 100  
    
    full_row_ptr = [0]
    full_col_idx = []
    part_row_ptr = [0]
    part_col_idx = []
    load_row_ptr = [0]
    load_col_idx = []
    all_inner_bitmaps = []
    
    full_block_count = 0
    part_block_count = 0
    load_block_count = 0
    
    for b in range(batch_size):
        for i in range(0, n, block_size_m):
            for j in range(0, n, block_size_n):
                outer_tile = Mask[b, i:i+block_size_m, j:j+block_size_n]
                
                if paddle.all(outer_tile == 1):
                    full_col_idx.append(j // block_size_n)
                    full_block_count += 1
                elif paddle.all(outer_tile == 0):
                    continue
                else:
                    part_col_idx.append(j // block_size_n)
                    part_block_count += 1
                    inner_bitmaps = get_InnerTile_bitmap(outer_tile.numpy())
                    all_inner_bitmaps.extend(inner_bitmaps)
                
                load_col_idx.append(j // block_size_n)
                load_block_count += 1
            
            full_row_ptr.append(full_block_count)
            part_row_ptr.append(part_block_count)
            load_row_ptr.append(load_block_count)
    
    device = Mask.place
    full_row_ptr = paddle.to_tensor(full_row_ptr, dtype='int32', place=device)
    full_col_idx = paddle.to_tensor(full_col_idx, dtype='int32', place=device)
    part_row_ptr = paddle.to_tensor(part_row_ptr, dtype='int32', place=device)
    part_col_idx = paddle.to_tensor(part_col_idx, dtype='int32', place=device)
    load_row_ptr = paddle.to_tensor(load_row_ptr, dtype='int32', place=device)
    load_col_idx = paddle.to_tensor(load_col_idx, dtype='int32', place=device)
    
    inner_bitmaps_tensor = paddle.to_tensor(
        [int(x) for x in all_inner_bitmaps],
        dtype='int64',
        place=device
    )
    
    return nnz, full_row_ptr, full_col_idx, part_row_ptr, part_col_idx, load_row_ptr, load_col_idx, inner_bitmaps_tensor

if __name__ == "__main__":
    paddle.seed(0)
    running_device = paddle_device_identify(print_info=True)

    parser = argparse.ArgumentParser(description="Attention test with Mask")
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--head_num', type=int, default=1)
    parser.add_argument('--head_size', type=int, default=64)
    parser.add_argument('--seq_len', type=int, default=128)
    
    args = parser.parse_args()

    batch_size = args.batch_size
    nheads = args.head_num
    headdim = args.head_size
    seqlen = args.seq_len
    dropout_p = 0.0
    dtype = 'float16'
    is_causal = True

    avg_seq_len = seqlen
    low, high = (2 * avg_seq_len - seqlen, seqlen + 1)
    input_lens = paddle.randint(low=low, high=high, shape=[batch_size])
    seqlen_mask = seqlen_to_mask(input_lens, seqlen)
    attr_mask = set_dtype(paddle.tile(seqlen_mask, repeat_times=[seqlen]).reshape([batch_size, seqlen, seqlen]), "fp16")
    
    if is_causal:
        mask_name = 'Causal_Mask'
        mask = generate_causal_mask(attr_mask)
    else:
        mask_name = 'Full_Mask'

    q = paddle.randn([batch_size, seqlen, nheads, headdim], dtype=dtype)
    k = paddle.randn([batch_size, seqlen, nheads, headdim], dtype=dtype)
    v = paddle.randn([batch_size, seqlen, nheads, headdim], dtype=dtype)
    
    print(f"q (batch_size, seqlen, nheads, head_dim) = {q.shape}")
    print(f"mask (batch_size, seqlen, seqlen) = {mask.shape}\n")
    
    pd_out = attention_paddle(q, k, v, dropout_p=dropout_p, causal=is_causal)
    
    print(pd_out)
    
    BLOCK_M = 64
    BLOCK_N = 64
    num_warps = 1
    
    nnz, full_row_ptr, full_col_idx, part_row_ptr, part_col_idx, load_row_ptr, load_col_idx, inner_bitmaps = get_OuterTile_storage(mask, BLOCK_M, BLOCK_N)

    binding_out = custom_attention(
        q, k, v,
        full_row_ptr, full_col_idx,
        part_row_ptr, part_col_idx, inner_bitmaps,
        load_row_ptr, load_col_idx,
        p_dropout=dropout_p,
        softmax_scale=1.0 / math.sqrt(headdim),
        is_causal=is_causal,
        window_size_left=-1,
        window_size_right=0,
        softcap=0.0,
        return_softmax=False
    )