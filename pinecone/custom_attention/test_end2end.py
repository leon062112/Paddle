import argparse
import paddle
import math
import warnings
from util.utils import set_dtype, seqlen_to_mask, paddle_device_identify, time_stamp_cudasync
from util.masks import generate_causal_mask, get_OuterTile_storage
from custom_attention import custom_attention
import paddle.incubate.cc as pcc
import paddle.incubate.cc.typing as pct
warnings.filterwarnings("ignore")
import os

def ref_program(Q, K, V, is_causal):
    dim = Q.size(-1)
    scores = paddle.einsum('bhqd,bhkd->bhqk', Q, K)
    scores = scores / paddle.sqrt(paddle.to_tensor([dim], dtype=scores.dtype))
    if is_causal:
        seq_q = Q.size(2)
        seq_kv = K.size(2)
        mask = paddle.tril(paddle.ones([seq_q, seq_kv], dtype=scores.dtype))
        mask = mask.unsqueeze(0).unsqueeze(0)
        scores = paddle.where(mask == 0, paddle.full_like(scores, float('-inf')), scores)
    attention_weights = paddle.nn.functional.softmax(scores, axis=-1)
    output = paddle.einsum('bhqk,bhkd->bhqd', attention_weights, V)
    return output

def transpose_for_scores(x, num_heads, head_size):
    """将 tensor 转换为 attention score 计算所需的 shape: [batch, seq_len, hidden] -> [batch, seq_len, num_heads, head_size]"""
    batch_size, seq_len, hidden_dim = x.shape
    x = x.reshape([batch_size, seq_len, num_heads, head_size])
    return x

# AP 融合函数：matmul + add + gelu
# 实际 shapes: x=[B, M, K], kernel=[K, N], bias=[N]
# 其中 B=batch, M=seq_len, K=hidden_dim, N=hidden_dim*4
B = pct.DimVar(32)
M = pct.DimVar(128)
K = pct.DimVar(512)
N = pct.DimVar(2048)
DType = pct.DTypeVar("T", "float16")

def fused_matmul_add_gelu(
    x: pct.Tensor([B, M, K], DType),
    kernel: pct.Tensor([K, N], DType),
    bias: pct.Tensor([N], DType),  # bias 是 1D [N]，会自动 broadcast
):
    out = paddle.matmul(x, kernel)
    out = out + bias
    out = paddle.nn.functional.gelu(out)
    return out

# 编译融合函数
ap_path = f"{os.path.dirname(paddle.__file__)}/apy/matmul_pass"

fused_ffn1 = pcc.compile(fused_matmul_add_gelu, ap_path=ap_path)
USE_AP_FUSION = True
print("[AP] Fusion matmul+add+gelu compiled successfully")

def bert_fwd_std(mask):
    with paddle.no_grad():
        hidden_states = input_from_tensor
        for layer in range(layer_num):
            input_tensor = hidden_states

            qkv = qkv_kernel[layer] + qkv_bias[layer]
            q1, k1, v1 = qkv.chunk(3, axis=-1)
            q = transpose_for_scores(q1, head_num, head_size)
            k = transpose_for_scores(k1, head_num, head_size)
            v = transpose_for_scores(v1, head_num, head_size)
            # ------------------------------------------------------------- Attention start
            h_list = custom_attention(q, k, v,
                            full_row_ptr, full_col_idx,
                            part_row_ptr, part_col_idx, inner_bitmaps,
                            load_row_ptr, load_col_idx,
                            p_dropout=dropout_p,
                            softmax_scale=1.0 / math.sqrt(head_size),
                            is_causal=is_causal,
                            window_size_left=-1,
                            window_size_right=0,
                            softcap=0.0,
                            return_softmax=False)

            h = h_list[0]  # 取第一个输出 out
            # h.shape = [batch, seq_len, num_heads, head_size]
            # 转换为 [batch, seq_len, hidden_dim]
            new_context_layer_shape = [h.shape[0], h.shape[1], hidden_dim]
            hidden_states = h.reshape(new_context_layer_shape)                
            # ------------------------------------------------------------ Attention End
            hidden_states = paddle.matmul(hidden_states, attr_output_kernel[layer]) + attr_output_bias[layer]
            hidden_states = hidden_states + input_tensor
            hidden_states = paddle.nn.functional.layer_norm(hidden_states, normalized_shape=[hidden_dim],
                                        weight=attr_output_layernorm_gamma[layer], bias=attr_output_layernorm_beta[layer])
            residual = hidden_states
        
            # 使用 AP 融合算子：matmul + add + gelu
            if USE_AP_FUSION:
                hidden_states = fused_ffn1(hidden_states, inter_kernel[layer], inter_bias[layer])
            else:
                hidden_states = paddle.matmul(hidden_states, inter_kernel[layer]) + inter_bias[layer]
                hidden_states = paddle.nn.functional.gelu(hidden_states)
            hidden_states = paddle.matmul(hidden_states, output_kernel[layer]) + output_bias[layer]
            hidden_states = hidden_states + residual 
            hidden_states = paddle.nn.functional.layer_norm(hidden_states, normalized_shape=[hidden_dim],  
                                        weight=output_layernorm_gamma[layer], bias=output_layernorm_beta[layer])  
        
            transformer_output[layer] = hidden_states

if __name__ == "__main__":
    paddle.seed(0)
    paddle.device.cuda.empty_cache()
    device = paddle_device_identify(print_info=True)
    
    is_A100 = True
    
    parser = argparse.ArgumentParser(description="Give the parameters for the attention test (with Mask)")
    parser.add_argument('--mask_id', type=int, default=0, help='Mask type: 1-Sliding | 2-Longformer | 3-BigBird (default: 0)')
    parser.add_argument('--block_m', type=int, default=64, help='Block Size of M (default:64)')
    parser.add_argument('--block_n', type=int, default=64, help='Block Size of N (default:64)')
    parser.add_argument('--num_warps', type=int, default=1, help='Warp Num to launch (default:4)')
    
    parser.add_argument('--method', type=str, default="PaddleNative", help='PaddleNative, STOF')
    parser.add_argument('--model', type=str, default="bert_base", help='Sequence length (default: 1)')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size (default: 1)')
    parser.add_argument('--seq_len', type=int, default=128, help='Sequence length (default: 256)')
    args = parser.parse_args() 

    mask_id    = args.mask_id
    BLOCK_M    = args.block_m
    BLOCK_N    = args.block_n
    num_warps  = args.num_warps
    model_selection  = args.model
    method_selection = args.method
    
    head_size = 64
    seq_len   = args.seq_len
    batch_size = args.batch_size
    
    data_type  = 'float16'
    dtype = "fp16"
    running_device = "gpu"
    attention_type = "paddle_attention"

    if num_warps > (BLOCK_M // 16) * (BLOCK_N // 16):
        print(f"num_warps: {num_warps}, (BLOCK_M/16) * (BLOCK_N/16): {int(BLOCK_M / 16) * int(BLOCK_N / 16)}")
        print("Error! Here should be: num_warps <= (BLOCK_M/16) * (BLOCK_N/16) !")
        exit(0)
               
    running_device = "gpu"
    sqrt_seq_len = int(math.sqrt(seq_len))
    fill_rate    = 0.1
    dropout_p = 0.0
    layer_num = 1
    num_channels = 3
    
    warmup_iters = 10
    running_iters = 10
    head_num = 8
    head_size = 64
    # 始终定义 inference_model
    inference_model = bert_fwd_std
    if model_selection == "bert_small":
        layer_num = 1
    else:
        layer_num = 1
    hidden_dim = head_num * head_size

    noattention = paddle.randn([batch_size, seq_len, head_num, head_size], dtype='float16')

    test_Paddle = False
    test_STOF = False
    test_RefProgram = False
    
    if method_selection == "PaddleNative":
        test_Paddle = True
    elif method_selection == "STOF":
        test_STOF = True

    
    avg_seq_len = seq_len 
    low, high = (2 * avg_seq_len - seq_len, seq_len + 1)
    input_lens = paddle.randint(low=low, high=high + 1, shape=[batch_size])
    seqlen_mask = seqlen_to_mask(input_lens, seq_len)
    attr_mask = set_dtype(paddle.tile(seqlen_mask, repeat_times=[seq_len]).reshape([batch_size, seq_len, seq_len]), "fp16")
    
    is_causal = True
    mask_name = 'Causal_Mask'
    mask = generate_causal_mask(attr_mask)
    
    qkv_kernel_raw              = [set_dtype(paddle.zeros([hidden_dim, hidden_dim * 3]).uniform_(-0.4, 0.4), dtype) for _ in range(layer_num)]
    attr_output_kernel          = [set_dtype(paddle.zeros([hidden_dim, hidden_dim]).uniform_(-0.4, 0.4), dtype) for _ in range(layer_num)]
    attr_output_bias            = [set_dtype(paddle.zeros([hidden_dim]).uniform_(-0.4, 0.4), dtype) for _ in range(layer_num)]
    attr_output_layernorm_gamma = [set_dtype(paddle.zeros([hidden_dim]).uniform_(-0.4, 0.4), dtype) for _ in range(layer_num)]
    attr_output_layernorm_beta  = [set_dtype(paddle.zeros([hidden_dim]).uniform_(-0.4, 0.4), dtype) for _ in range(layer_num)]
    inter_kernel                = [set_dtype(paddle.zeros([hidden_dim, hidden_dim * 4]).uniform_(-0.4, 0.4), dtype) for _ in range(layer_num)]
    inter_bias                  = [set_dtype(paddle.zeros([hidden_dim * 4]).uniform_(-0.4, 0.4), dtype) for _ in range(layer_num)]
    output_kernel               = [set_dtype(paddle.zeros([hidden_dim * 4, hidden_dim]).uniform_(-0.4, 0.4), dtype) for _ in range(layer_num)]
    output_bias                 = [set_dtype(paddle.zeros([hidden_dim]).uniform_(-0.4, 0.4), dtype) for _ in range(layer_num)]
    output_layernorm_gamma      = [set_dtype(paddle.zeros([hidden_dim]).uniform_(-0.4, 0.4), dtype) for _ in range(layer_num)]
    output_layernorm_beta       = [set_dtype(paddle.zeros([hidden_dim]).uniform_(-0.4, 0.4), dtype) for _ in range(layer_num)]
    
    input_from_tensor           = set_dtype(paddle.empty([batch_size, seq_len, hidden_dim]).uniform_(-0.4, 0.4), dtype)
    qkv_bias                    = [set_dtype(paddle.zeros([hidden_dim * 3]).uniform_(-0.4, 0.4), dtype) for _ in range(layer_num)]
    qkv_kernel                  = [set_dtype(paddle.zeros([batch_size, seq_len, hidden_dim * 3]).uniform_(-0.4, 0.4), dtype) for _ in range(layer_num)]
   
    transformer_output = [None for _ in range(layer_num)]

    nnz, full_row_ptr, full_col_idx, part_row_ptr, part_col_idx, load_row_ptr, load_col_idx, inner_bitmaps = get_OuterTile_storage(mask, BLOCK_M, BLOCK_N)    

    # Paddle Naive  ---------------------------------------
    if test_Paddle:
        attention_type = "paddle_attention"
        
        for i in range(warmup_iters + running_iters):
            if i == warmup_iters:    
                t1_start = time_stamp_cudasync()

            output = ref_program(noattention, noattention, noattention, is_causal)
            paddle_compiled_output = output

        t1_end = time_stamp_cudasync()
        paddle_compiled_time = (t1_end - t1_start) * 1000 / running_iters   
        print("e2e {} | bs:{} | seq:{} | Paddle Native   : {:.3f} ms / iter".format(
            model_selection, batch_size, args.seq_len, paddle_compiled_time)) 

    #  STOF_attention ------------------------------------
    if test_STOF:    
        attention_type = "STOF_attention"
        for i in range(warmup_iters + running_iters):
            if i == warmup_iters:    
                t_start = time_stamp_cudasync()

            inference_model(mask)
            STOF_output = transformer_output[-1]
        
        t_end = time_stamp_cudasync()
        STOF_time = (t_end - t_start) * 1000 / running_iters
        print("e2e {} | bs:{} | seq:{}  |  STOF            : {:.3f} ms / iter".format(
            model_selection, batch_size, args.seq_len, STOF_time)) 
