"""Copyright (c) 2026, Dao AI Lab, Goombalab. Apache-2.0; see LICENSE.
Modified by Continuous Intelligence: extracted forward-only inference utilities.
"""

import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({}, num_stages=s, num_warps=w)
        for s in [2, 3]
        for w in [4, 8]
    ],
    key=["CHUNK_SIZE"],
)
@triton.jit
def dacs_segsum_kernel(
    da_ptr,
    da_cs_ptr,
    da_cs_rev_ptr,
    segsum_ptr,
    stride_da_batch, stride_da_head, stride_da_seq,
    stride_da_cs_batch, stride_da_cs_head, stride_da_cs_seq,
    stride_da_cs_rev_batch, stride_da_cs_rev_head, stride_da_cs_rev_seq,
    stride_segsum_batch, stride_segsum_head, stride_segsum_chunk,
    stride_segsum_row, stride_segsum_col,
    SEQLEN: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
):
    """Dense (fixed-length) version of dacs_segsum_kernel_varlen.

    Computes da_cs, da_cs_rev, and the lower-triangular segsum matrix for each
    contiguous chunk.  Sequences are assumed to be fixed-length and aligned to
    chunk boundaries; no varlen mapping tensors are needed.

    Grid: (B, H, nchunks).
    """
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_chunk = tl.program_id(2)

    chunk_start = pid_chunk * CHUNK_SIZE
    offs = tl.arange(0, CHUNK_SIZE)
    offs_seq = chunk_start + offs
    mask = offs_seq < SEQLEN

    base_da = pid_batch * stride_da_batch + pid_head * stride_da_head
    da_chunk = tl.load(da_ptr + base_da + offs_seq * stride_da_seq, mask=mask, other=0.0)

    da_cs = tl.cumsum(da_chunk, axis=0)
    da_cs = tl.minimum(da_cs, 0.0)

    da_cs_rev = tl.cumsum(da_chunk, axis=0, reverse=True)
    # Roll one to the left:
    i = tl.arange(0, CHUNK_SIZE)[:, None]          # [N,1]
    j = tl.arange(0, CHUNK_SIZE)[None, :]          # [1,N]
    S_mat = (i == j - 1)                      # strictly upper diagonal (one above main)
    da_cs_rev = tl.sum(tl.where(S_mat, da_cs_rev, 0), axis=1)
    da_cs_rev = tl.minimum(da_cs_rev, 0.0)

    base_da_cs = pid_batch * stride_da_cs_batch + pid_head * stride_da_cs_head
    base_da_cs_rev = pid_batch * stride_da_cs_rev_batch + pid_head * stride_da_cs_rev_head
    tl.store(da_cs_ptr + base_da_cs + offs_seq * stride_da_cs_seq, da_cs, mask=mask)
    tl.store(da_cs_rev_ptr + base_da_cs_rev + offs_seq * stride_da_cs_rev_seq, da_cs_rev, mask=mask)

    broadcasted_indices = tl.zeros_like(offs)
    segsum = tl.load(da_ptr + base_da + offs_seq[:, None] * stride_da_seq + broadcasted_indices[None, :])
    offs_i = offs[:, None]
    offs_j = offs[None, :]
    segsum = tl.where(offs_i > offs_j, segsum, 0.0)
    segsum = tl.cumsum(segsum, axis=0)
    segsum = tl.minimum(segsum, 0.0)

    base_segsum = (pid_batch * stride_segsum_batch +
                   pid_head * stride_segsum_head +
                   pid_chunk * stride_segsum_chunk)
    tl.store(segsum_ptr + base_segsum + offs_i * stride_segsum_row + offs_j * stride_segsum_col, segsum)


def compute_dacs_segsum_triton(
    da: torch.Tensor,  # [B, H, S]
    chunk_size: int,
):
    """Compute da_cs, da_cs_rev, and segsum for fixed-length sequences.

    Returns
    -------
    da_cs    : forward inclusive prefix-sum of da, clipped to ≤ 0. [B, H, S]
    da_cs_rev: exclusive reverse prefix-sum of da, clipped to ≤ 0. [B, H, S]
    segsum   : lower-triangular intra-chunk segsum. [B, H, nchunks, C, C]
    """
    B, H, S = da.shape
    nchunks = (S + chunk_size - 1) // chunk_size

    da_cs = torch.empty_like(da)
    da_cs_rev = torch.empty_like(da)
    segsum = torch.empty(B, H, nchunks, chunk_size, chunk_size, device=da.device, dtype=da.dtype)

    grid = (B, H, nchunks)
    dacs_segsum_kernel[grid](
        da, da_cs, da_cs_rev, segsum,
        da.stride(0), da.stride(1), da.stride(2),
        da_cs.stride(0), da_cs.stride(1), da_cs.stride(2),
        da_cs_rev.stride(0), da_cs_rev.stride(1), da_cs_rev.stride(2),
        segsum.stride(0), segsum.stride(1), segsum.stride(2),
        segsum.stride(3), segsum.stride(4),
        S,
        chunk_size,
    )

    return da_cs, da_cs_rev, segsum
