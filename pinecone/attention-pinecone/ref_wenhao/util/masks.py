import math
import paddle
import random

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

def generate_triangle_mask(attr_mask):
    """Generate lower triangular mask.
    
    Args:
        attr_mask (paddle.Tensor): Input attention mask.
    
    Returns:
        paddle.Tensor: Lower triangular mask.
    """
    seq_len = attr_mask.shape[1]
    triangle_mask = paddle.tril(paddle.ones([seq_len, seq_len]))
    return triangle_mask.unsqueeze(0).expand([attr_mask.shape[0], -1, -1])

def generate_strided_mask(attr_mask):
    """Generate strided attention mask.
    
    Args:
        attr_mask (paddle.Tensor): Input attention mask.
    
    Returns:
        paddle.Tensor: Strided mask tensor.
    """
    stride_step = int(math.sqrt(attr_mask.shape[1]))
    seq_len = attr_mask.shape[1]
    strided_mask = paddle.zeros_like(attr_mask)
    
    for batch in range(strided_mask.shape[0]):
        for i in range(seq_len):
            for j in range(i+1):
                if (i - j) % stride_step == 0 or j > i - stride_step:
                    strided_mask[batch, i, j] = 1.0
    return strided_mask

def generate_fixed_mask(attr_mask):
    """Generate fixed pattern attention mask.
    
    Args:
        attr_mask (paddle.Tensor): Input attention mask.
    
    Returns:
        paddle.Tensor: Fixed pattern mask tensor.
    """
    fixed_step = int(math.sqrt(attr_mask.shape[1]))
    seq_len = attr_mask.shape[1]
    fixed_mask = paddle.zeros_like(attr_mask)
    
    for batch in range(fixed_mask.shape[0]):
        for i in range(seq_len):
            for j in range(i+1):
                if j % fixed_step == fixed_step-1 or j > i + (j % fixed_step) - fixed_step:
                    fixed_mask[batch, i, j] = 1.0
    return fixed_mask

def generate_full_mask(attr_mask):
    """Generate full attention mask.
    
    Args:
        attr_mask (paddle.Tensor): Input attention mask.
    
    Returns:
        paddle.Tensor: Full mask tensor with all ones.
    """
    return paddle.ones_like(attr_mask)

def generate_causal_mask(attr_mask):
    """Generate causal attention mask.
    
    Args:
        attr_mask (paddle.Tensor): Input attention mask.
    
    Returns:
        paddle.Tensor: Causal mask tensor.
    """
    seq_len = attr_mask.shape[1]
    causal_mask = paddle.tril(paddle.ones([seq_len, seq_len]))
    return causal_mask.unsqueeze(0).expand([attr_mask.shape[0], -1, -1])

def generate_sliding_mask(attr_mask, bandwidth=1, is_causal=False):
    """Generate sliding window attention mask.
    
    Args:
        attr_mask (paddle.Tensor): Input attention mask.
        bandwidth (int, optional): Window size. Defaults to 1.
        is_causal (bool, optional): Whether to apply causal masking. Defaults to False.
    
    Returns:
        paddle.Tensor: Sliding window mask tensor.
    """
    batch_size = attr_mask.shape[0]
    seq_len = attr_mask.shape[1]
    sliding_mask = paddle.zeros_like(attr_mask)

    for batch in range(batch_size):
        for i in range(seq_len):
            start = max(0, i - bandwidth)
            end = min(seq_len, i + bandwidth + 1)
            sliding_mask[batch, i, start:end] = 1
    
    if is_causal:
        for i in range(seq_len-1):
            sliding_mask[batch, i, i+1:] = 0

    return sliding_mask

def generate_dilated_mask(attr_mask, bandwidth=1, dilation_rate=1, is_causal=False):
    """Generate dilated attention mask.
    
    Args:
        attr_mask (paddle.Tensor): Input attention mask.
        bandwidth (int, optional): Base window size. Defaults to 1.
        dilation_rate (int, optional): Dilation rate. Defaults to 1.
        is_causal (bool, optional): Whether to apply causal masking. Defaults to False.
    
    Returns:
        paddle.Tensor: Dilated mask tensor.
    """
    batch_size = attr_mask.shape[0]
    seq_len = attr_mask.shape[1]
    dilated_mask = paddle.zeros_like(attr_mask)

    for batch in range(batch_size):
        for i in range(seq_len):
            start = i - bandwidth - dilation_rate
            end = min(seq_len, i + bandwidth + dilation_rate + 1)
            for row_idx in range(start, end, dilation_rate + 1):
                if row_idx > -1:
                    dilated_mask[batch, i, row_idx] = 1
    
    if is_causal:
        for i in range(seq_len-1):
            dilated_mask[batch, i, i+1:] = 0
            
    return dilated_mask

def generate_longformer_mask(attr_mask, globalwidth=1, bandwidth=1, is_causal=False):
    """Generate Longformer-style attention mask.
    
    Args:
        attr_mask (paddle.Tensor): Input attention mask.
        globalwidth (int, optional): Global attention width. Defaults to 1.
        bandwidth (int, optional): Local window size. Defaults to 1.
        is_causal (bool, optional): Whether to apply causal masking. Defaults to False.
    
    Returns:
        paddle.Tensor: Longformer mask tensor.
    """
    batch_size = attr_mask.shape[0]
    seq_len = attr_mask.shape[1]
    longformer_mask = paddle.zeros_like(attr_mask)
    
    # Global attention bands
    longformer_mask[:, :globalwidth, :] = 1
    longformer_mask[:, :, :globalwidth] = 1
    
    center = seq_len // 2
    longformer_mask[:, center:center+globalwidth, :] = 1
    longformer_mask[:, :, center:center+globalwidth] = 1

    # Local window attention
    for batch in range(batch_size):
        for i in range(seq_len):
            start = max(0, i - bandwidth)
            end = min(seq_len, i + bandwidth + 1)
            longformer_mask[batch, i, start:end] = 1
    
    if is_causal:
        for i in range(seq_len-1):
            longformer_mask[:, i, i+1:] = 0

    return longformer_mask

def generate_bigbird_mask(attr_mask, globalwidth=1, bandwidth=1, fill_rate=0.2, is_causal=False):
    """Generate BigBird-style attention mask.
    
    Args:
        attr_mask (paddle.Tensor): Input attention mask.
        globalwidth (int, optional): Global attention width. Defaults to 1.
        bandwidth (int, optional): Local window size. Defaults to 1.
        fill_rate (float, optional): Random attention density. Defaults to 0.2.
        is_causal (bool, optional): Whether to apply causal masking. Defaults to False.
    
    Returns:
        paddle.Tensor: BigBird mask tensor.
    """
    batch_size = attr_mask.shape[0]
    seq_len = attr_mask.shape[1]
    bigbird_mask = paddle.zeros_like(attr_mask)
    
    # Global attention bands
    bigbird_mask[:, :globalwidth, :] = 1
    bigbird_mask[:, :, :globalwidth] = 1
    
    # Local window attention
    for batch in range(batch_size):
        for i in range(seq_len):
            start = max(0, i - bandwidth)
            end = min(seq_len, i + bandwidth + 1)
            bigbird_mask[batch, i, start:end] = 1
    
    # Random attention
    num_ones_in_block = int(1024 * fill_rate)
    random.seed(0)
    
    for batch in range(batch_size):
        for i in range(seq_len // 32 - 1):
            start_x = i * 32
            if i == 0:
                start_y = start_x + 32 * random.choice([1, 0])
            elif i == seq_len // 32 - 2:
                start_y = start_x + 32 * random.choice([-1, 0])
            else:
                start_y = start_x + 32 * random.choice([-1, 0, 1])
            
            for _ in range(num_ones_in_block):
                random_x = random.randint(0, 32)
                random_y = random.randint(0, 32)
                bigbird_mask[batch, start_x + random_x, start_y + random_y] = 1
    
    if is_causal:
        for i in range(seq_len-1):
            bigbird_mask[:, i, i+1:] = 0

    return bigbird_mask