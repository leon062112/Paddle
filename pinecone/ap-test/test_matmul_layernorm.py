# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import subprocess
import unittest

import numpy as np

import paddle
import paddle.incubate.cc as pcc
import paddle.incubate.cc.typing as pct

os.environ["AP_WORKSPACE_DIR"] = "/tmp/paddle/ap"


def GetPirProgram(fused_func, tensor_args):
    dtypes = tuple(tensor.dtype for tensor in tensor_args)
    func = fused_func.func_overload_ctx.dtypes2func.get(dtypes, None)
    return str(func.infer_program.forward_program)


def IsCertainDevices():
    try:
        sp = subprocess.Popen(
            ['nvidia-smi', '-q'], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        out_str = sp.communicate()[0].decode('utf-8')
        if 'A100' in out_str:
            return True
        else:
            return False
    except Exception as e:
        return False


class TestMatmulLayerNorm(unittest.TestCase):
    def setUp(self):
        dtype = 'float16'
        x_shape = [32, 16, 16]      # [B, M, K]
        self.x = paddle.randn(x_shape, dtype=dtype)
        self.x.stop_gradient = False

        w_shape = [16, 32]          # [K, N] → 注意：这里 N=32，让 normalized_shape=[32]
        self.w = paddle.randn(w_shape, dtype=dtype)
        self.w.stop_gradient = False

        # LayerNorm 的 normalized_shape 必须匹配matmul输出的尾部维度
        # matmul(x, w): [32,16,16] × [16,32] → [32,16,32]
        self.normalized_shape = [32]

    def getSubGraph(self):
        B = pct.DimVar(32)
        M = pct.DimVar(16)
        K = pct.DimVar(16)
        N = pct.DimVar(32)          # 注意：N=32，与 w_shape[1] 一致
        DType = pct.DTypeVar("T", "float16")

        def foo(
            x: pct.Tensor([B, M, K], DType),
            w: pct.Tensor([K, N], DType),
        ):
            y = paddle.matmul(x, w)  # [B, M, N]
            out = paddle.nn.functional.layer_norm(
                y, normalized_shape=self.normalized_shape
            )
            return out

        return foo

    def test_subgraph(self):
        foo = self.getSubGraph()
        fused_foo = pcc.compile(
            foo, ap_path=f"{os.path.dirname(paddle.__file__)}/apy/matmul__pass"
        )
        generated_pir_program = GetPirProgram(fused_foo, [self.x, self.w])
        self.assertTrue(
            'pd_op.ap_variadic' in generated_pir_program, "fusion failed"
        )
        if IsCertainDevices():
            ap_outs = fused_foo(self.x, self.w)
            dy_outs = foo(self.x, self.w)
            np.testing.assert_allclose(dy_outs, ap_outs, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    unittest.main()