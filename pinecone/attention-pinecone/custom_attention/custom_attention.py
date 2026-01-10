from paddle import _C_ops
from paddle.framework import in_dynamic_or_pir_mode
from paddle.base.layer_helper import LayerHelper
from paddle.jit.marker import unified

@unified
def custom_attention(q,k,v,full_row_ptr,full_col_idx,part_row_ptr,part_col_idx,inner_bitmaps,load_row_ptr,load_col_idx,p_dropout,softmax_scale,is_causal,window_size_left,window_size_right,softcap,return_softmax):
    # The output variable's dtype use default value 'float32',
    # and the actual dtype of output variable will be inferred in runtime.
    if in_dynamic_or_pir_mode():
        outs = _C_ops._run_custom_op("custom_attention", q,k,v,full_row_ptr,full_col_idx,part_row_ptr,part_col_idx,inner_bitmaps,load_row_ptr,load_col_idx,p_dropout,softmax_scale,is_causal,window_size_left,window_size_right,softcap,return_softmax)
        res = []
        start_idx = 0
        res.append(outs[start_idx])
        start_idx += 1
        res.append(outs[start_idx])
        start_idx += 1
        res.append(outs[start_idx])
        start_idx += 1
        res.append(outs[start_idx])
        start_idx += 1
        return res[0] if len(res)==1 else res
    else:
        ins = {}
        ins_map = {'q' : q,'k' : k,'v' : v,'full_row_ptr' : full_row_ptr,'full_col_idx' : full_col_idx,'part_row_ptr' : part_row_ptr,'part_col_idx' : part_col_idx,'inner_bitmaps' : inner_bitmaps,'load_row_ptr' : load_row_ptr,'load_col_idx' : load_col_idx}
        outs = {}
        outs_list = ['out','softmax_lse','p','rng_state']
        for key, value in ins_map.items():
            # handle optional inputs
            if value is not None:
                ins[key] = value
        helper = LayerHelper("custom_attention", **locals())

        outs['out'] = helper.create_variable(dtype='float32')
        outs['softmax_lse'] = helper.create_variable(dtype='float32')
        outs['p'] = helper.create_variable(dtype='float32')
        outs['rng_state'] = helper.create_variable(dtype='float32')
        helper.append_op(type="custom_attention", inputs=ins, outputs=outs, attrs={'p_dropout' : p_dropout,'softmax_scale' : softmax_scale,'is_causal' : is_causal,'window_size_left' : window_size_left,'window_size_right' : window_size_right,'softcap' : softcap,'return_softmax' : return_softmax})
        res = [outs[out_name] if out_name in outs.keys() else None for out_name in outs_list]
        return res[0] if len(res)==1 else res


import os
import sys
import types
import paddle
import importlib.abc
import importlib.util

cur_dir = os.path.dirname(os.path.abspath(__file__))
so_path = os.path.join(cur_dir, "custom_attention_pd_.so")

def __bootstrap__():
    assert os.path.exists(so_path)
    # load custom op shared library with abs path
    custom_ops = paddle.utils.cpp_extension.load_op_meta_info_and_register_op(so_path)

    if os.name == 'nt' or sys.platform.startswith('darwin'):
        # Cpp Extension only support Linux now
        mod = types.ModuleType(__name__)
    else:
        try:
            spec = importlib.util.spec_from_file_location(__name__, so_path)
            assert spec is not None
            mod = importlib.util.module_from_spec(spec)
            assert isinstance(spec.loader, importlib.abc.Loader)
            spec.loader.exec_module(mod)
        except ImportError:
            mod = types.ModuleType(__name__)

    for custom_op in custom_ops:
        setattr(mod, custom_op, eval(custom_op))

__bootstrap__()

