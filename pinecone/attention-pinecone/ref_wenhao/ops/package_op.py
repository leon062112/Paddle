import paddle
from paddle.autograd import PyLayer
from typing import Optional, Sequence, Tuple, Union

import binding_attn

__all__ = ['binding_attn_func']

def _binding_attn_forward(
    q: paddle.Tensor, k: paddle.Tensor, v: paddle.Tensor,
    full_row_ptr: paddle.Tensor, full_col_idx: paddle.Tensor,
    part_row_ptr: paddle.Tensor, part_col_idx: paddle.Tensor, 
    inner_bitmaps: paddle.Tensor,
    load_row_ptr: paddle.Tensor, load_col_idx: paddle.Tensor,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int, window_size_right: int,
    softcap: float
) -> Tuple[paddle.Tensor, paddle.Tensor]:
    
    maybe_contiguous = lambda x: x if x.is_contiguous() else x.contiguous()
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    
    out, softmax_lse = binding_attn.flash_attn_binding(
        q, k, v,
        full_row_ptr, full_col_idx,
        part_row_ptr, part_col_idx, inner_bitmaps,
        load_row_ptr, load_col_idx,
        p_dropout=dropout_p,
        softmax_scale=softmax_scale,
        is_causal=causal,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
        softcap=softcap
    )
    return out, softmax_lse

class BindingAttnFunc(PyLayer):
    @staticmethod
    def forward(ctx,
        q: paddle.Tensor, k: paddle.Tensor, v: paddle.Tensor,
        full_row_ptr: paddle.Tensor, full_col_idx: paddle.Tensor,
        part_row_ptr: paddle.Tensor, part_col_idx: paddle.Tensor,
        inner_bitmaps: paddle.Tensor,
        load_row_ptr: paddle.Tensor, load_col_idx: paddle.Tensor,
        dropout_p: float,
        softmax_scale: float,
        causal: bool,
        window_size: Tuple[int, int],
        softcap: float,
        return_attn_probs: bool
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
            
        head_size_og = q.shape[3]
        if head_size_og % 8 != 0:
            q = paddle.nn.functional.pad(q, [0, 8 - head_size_og % 8])
            k = paddle.nn.functional.pad(k, [0, 8 - head_size_og % 8])
            v = paddle.nn.functional.pad(v, [0, 8 - head_size_og % 8])
            
        out_padded, softmax_lse = _binding_attn_forward(
            q, k, v,
            full_row_ptr, full_col_idx,
            part_row_ptr, part_col_idx, inner_bitmaps,
            load_row_ptr, load_col_idx,
            dropout_p,
            softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            softcap=softcap
        )
        
        out = out_padded[..., :head_size_og]
        return out if not return_attn_probs else (out, softmax_lse)

def binding_attn_func(
    q, k, v,
    full_row_ptr, full_col_idx,
    part_row_ptr, part_col_idx, inner_bitmaps,
    load_row_ptr, load_col_idx,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    return_attn_probs=False
):
    return BindingAttnFunc.apply(
        q, k, v,
        full_row_ptr, full_col_idx,
        part_row_ptr, part_col_idx, inner_bitmaps,
        load_row_ptr, load_col_idx,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        return_attn_probs
    )