# attention_with_mask_ptn.py

from paddle.incubate.cc import pattern
from paddle.incubate.cc.pattern import OpPattern, TensorPattern, ConstantPattern
from paddle.incubate.cc.pattern import graph_pattern_matcher as gpm
from paddle.incubate.cc.pattern import register_pattern, register_rewriter

# Step 1: Define the source pattern
@register_pattern
class AttentionWithMaskPattern(OpPattern):
    def __init__(self):
        super().__init__(name="attention_with_mask")

    def build(self):
        # Input tensors
        q = TensorPattern()
        k = TensorPattern()
        v = TensorPattern()
        mask = TensorPattern()

        # k^T
        kt = self.transpose(k, perm=[0, 1, 3, 2])

        # matmul(q, kt)
        scores = self.matmul(q, kt)

        # scale: scores * 0.125
        scale_val = ConstantPattern(value=0.125)
        scaled_scores = self.multiply(scores, scale_val)

        # (1.0 - mask)
        one = ConstantPattern(value=1.0)
        one_minus_mask = self.subtract(one, mask)

        # 10000.0 * (1 - mask)
        large_val = ConstantPattern(value=10000.0)
        mask_penalty = self.multiply(large_val, one_minus_mask)

        # scaled_scores - mask_penalty
        masked_scores = self.subtract(scaled_scores, mask_penalty)

        # softmax
        probs = self.softmax(masked_scores, axis=-1)

        # final matmul: probs @ v
        output = self.matmul(probs, v)

        # Set inputs and output
        self.inputs = [q, k, v, mask]
        self.output = output
        return self

# Step 2: Define the rewriter that maps to SongguoAttn
@register_rewriter(pattern_name="attention_with_mask")
class AttentionWithMaskRewriter:
    def rewrite(self, op_graph, matched_ops):
        """
        matched_ops: dict of {pattern_node -> actual op}
        We extract q, k, v, mask from inputs.
        """
        # Get the input tensors from the matched pattern
        q = matched_ops["inputs"][0]
        k = matched_ops["inputs"][1]
        v = matched_ops["inputs"][2]
        mask = matched_ops["inputs"][3]

        # Create the new op: SongguoAttn(q, k, v, mask)
        # Note: Paddle will look for a registered op named "songguo_attn"
        songguo_attn_op = op_graph.add_op(
            op_name="songguo_attn",
            inputs={"Q": q, "K": k, "V": v, "Mask": mask},
            outputs={"Out": None},  # Let AP infer output
            attrs={}
        )

        # Replace the original output (the final matmul) with this new op's output
        original_output = matched_ops["output"]
        new_output = songguo_attn_op.outputs["Out"]

        # Replace in graph
        op_graph.replace_all_uses_with(original_output, new_output)

        # Remove the old subgraph (optional, AP may handle it)
        # For safety, we just return success
        return True