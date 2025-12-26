import os
import subprocess
import unittest
import time

import numpy as np

import paddle
import paddle.incubate.cc as pcc
import paddle.incubate.cc.typing as pct

import paddle.profiler as profiler

os.environ["AP_WORKSPACE_DIR"] = "/tmp/paddle/ap"

DT = 'float16'
BS = 4
MS = 128
NS = 32
KS = 128
# BS = 1
# MS = 3840
# NS = 4096
# KS = 4096
# BS = 1
# MS = 64
# NS = 3072
# KS = 768

class TestMatmulEpilogue(unittest.TestCase):
    def setUp(self):
        dtype = DT
        x_shape = [BS, MS, KS]
        self.x = paddle.ones(x_shape, dtype=dtype)
        self.x.stop_gradient = False

        y_shape = [KS, NS]
        self.y = paddle.ones(y_shape, dtype=dtype)
        self.y.stop_gradient = False

        b_shape = [MS, NS]
        self.b = paddle.ones(b_shape, dtype=dtype)
        self.b.stop_gradient = False

        b1_shape = [1]
        self.b1 = paddle.ones(b1_shape, dtype=dtype)
        self.b1.stop_gradient = False

    def getSubGraph(self):
        B = pct.DimVar(BS)
        M = pct.DimVar(MS)
        K = pct.DimVar(KS)
        N = pct.DimVar(NS)
        DType = pct.DTypeVar("T", DT)

        def foo(
            x: pct.Tensor([B, M, K], DType),
            y: pct.Tensor([K, N], DType),
            b: pct.Tensor([M, N], DType),
            b1: pct.Tensor([1], DType),
        ):

            out = paddle.matmul(x, y)
            out = out + b                       # [B, M, N] (broadcast add)
            return out
            # out = paddle.reshape(out, [-1, 256])  # reshape to [B*M, N]
            # return out + b
            # return paddle.multiply(out, b1)
            # return paddle.scale(out, scale=0.1)
            # return paddle.nn.functional.relu(out + b)
            # return paddle.cast(out + b, "float32")
            # return paddle.nn.functional.gelu(out + b)

            # out_fp32 = paddle.cast(out, "float32")
            # out_fp32 = out_fp32 + b1
            # return paddle.scale(out_fp32, scale=1.0)

            # return out

        return foo

    def test_subgraph(self):
        foo = self.getSubGraph()
        fused_foo = pcc.compile(
            foo, ap_path=f"{os.path.dirname(paddle.__file__)}/apy/matmul_pass"
        )

        ap_outs = fused_foo(self.x, self.y, self.b, self.b1)
        dy_outs = foo(self.x, self.y, self.b, self.b1)
        # print(f"ap_outs: {ap_outs}")
        # print(f"dy_outs: {dy_outs}")

        # -------- 性能测试部分 --------
        iters = 10
        # warmup
        _ = fused_foo(self.x, self.y, self.b, self.b1)
        _ = foo(self.x, self.y, self.b, self.b1)

        # paddle.device.synchronize()
        # start = time.time()
        # # for _ in range(iters):
        #     # _ = fused_foo(self.x, self.y, self.b, self.b1)
        # paddle.device.synchronize()
        # end = time.time()
        # avg_time = (end - start) / iters
        # print(f"[Performance] Avg latency per run: {avg_time:.6f} s")

        # profiler (保存到 log_dir)
        with profiler.Profiler(
            targets=[profiler.ProfilerTarget.CPU, profiler.ProfilerTarget.GPU],
            on_trace_ready=profiler.export_chrome_tracing("./profiler_log"),
            timer_only = False
        ) as prof:
            for step in range(iters):
                _ = fused_foo(self.x, self.y, self.b, self.b1)
                # _ = foo(self.x, self.y, self.b, self.b1)
                prof.step()
        print("[Profiler] Trace saved to ./profiler_log")
        prof.summary(sorted_by=profiler.SortedKeys.GPUTotal,
             op_detail=True,
             thread_sep=False,
             time_unit='us')

        for dy_out, ap_out in zip(dy_outs, ap_outs):
            np.testing.assert_allclose(dy_out, ap_out, atol=1e-1)

if __name__ == "__main__":
    unittest.main()
