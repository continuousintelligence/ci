"""Copyright (c) 2025, Dao AI Lab, Goombalab. Apache-2.0; see LICENSE.
Modified by Continuous Intelligence: extracted the inference utility needed by ci-proto.
"""

import triton
import triton.language as tl

@triton.jit
def tanh_approx(x):
    """
    (Fast) hyperbolic tangent approximation using PTX inline assembly.

    Args:
        x: Input triton tensor (any shape) in float32
    Returns:
        Approximate tanh values in float32
    """
    return tl.inline_asm_elementwise(
        "tanh.approx.f32 $0, $1;",
        constraints="=f,f",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
