import paddle
import math
import timeit
import re


def time_stamp_cudasync():
    paddle.device.cuda.synchronize()
    return timeit.default_timer() 

def paddle_device_identify(print_info=True):
    """Identify PaddlePaddle device information.
    
    Args:
        print_info (bool, optional): Whether to print device info. Defaults to True.
    
    Returns:
        paddle.Device: The current device.
    """
    if paddle.device.is_compiled_with_cuda():
        if print_info:
            print(' PaddlePaddle version:', paddle.__version__)
            print(' CUDA available \t:', paddle.device.cuda.device_count())
            print(' Current GPU \t: {}'.format(paddle.device.get_device()), 
                  '\n', "-" * 50)
        return paddle.device.get_device()
    else:
        print('CUDA is not available! Using CPU.')
        return paddle.CPUPlace()

def time_stamp_cuda_sync():
    """Get timestamp after synchronizing CUDA stream."""
    if paddle.device.is_compiled_with_cuda():
        paddle.device.cuda.synchronize()
    return timeit.default_timer()

def set_dtype(tensor, dtype):
    """Set tensor data type.
    
    Args:
        tensor (paddle.Tensor): Input tensor.
        dtype (str): Target dtype, supports 'fp32' or 'fp16'.
    
    Returns:
        paddle.Tensor: Tensor with new dtype.
    
    Raises:
        RuntimeError: If dtype is not supported.
    """
    if dtype == "fp32":
        return tensor.astype('float32')
    elif dtype == "fp16":
        return tensor.astype('float16')
    raise RuntimeError(f"Unsupported dtype {dtype}")

def transpose_for_scores(x, n_heads, head_size):
    """Transpose input tensor for multi-head attention scores.
    
    Args:
        x (paddle.Tensor): Input tensor with shape [batch_size, seq_len, hidden_dim].
        n_heads (int): Number of attention heads.
        head_size (int): Size of each attention head.
    
    Returns:
        paddle.Tensor: Transposed tensor with shape [batch_size, n_heads, seq_len, head_size].
    """
    new_shape = x.shape[:-1] + [n_heads, head_size]
    x = x.reshape(new_shape)
    return x.transpose([0, 2, 1, 3])

def seqlen_to_mask(lengths, max_len):
    """Generate sequence mask from lengths.
    
    Args:
        lengths (paddle.Tensor): Sequence lengths tensor.
        max_len (int): Maximum sequence length.
    
    Returns:
        paddle.Tensor: Generated mask tensor.
    """
    batch_size = lengths.size
    mask = (paddle.arange(0, max_len, dtype=lengths.dtype)
            .expand([batch_size, -1])
            .less_than(lengths.unsqueeze(1)))
    return mask