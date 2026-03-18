# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import mori
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.platforms import current_platform

logger = init_logger(__name__)

# Kernel launch configs tuned for prefill (large batch) vs decode (small batch).
# Prefill: more blocks + warps to saturate GPU with many tokens.
# Decode: fewer blocks + warps to reduce launch overhead with few tokens.
_PREFILL_BLOCK_NUM = 128
_PREFILL_WARP_PER_BLOCK = 16
_DECODE_BLOCK_NUM = 64
_DECODE_WARP_PER_BLOCK = 4
# Threshold: batches with more tokens than this use prefill config.
_PREFILL_TOKEN_THRESHOLD = 64


class MoriPrepareAndFinalize(mk.FusedMoEPrepareAndFinalizeModular):
    """
    Prepare/Finalize using MoRI kernels.
    """

    def __init__(
        self,
        mori_op: mori.ops.EpDispatchCombineOp,
        max_tokens_per_rank: int,
        num_dispatchers: int,
        use_fp8_dispatch: bool = False,
    ):
        super().__init__()
        self.mori_op = mori_op
        self.num_dispatchers_ = num_dispatchers
        self.max_tokens_per_rank = max_tokens_per_rank
        self.use_fp8_dispatch = use_fp8_dispatch

    @staticmethod
    def _launch_config(num_tokens: int) -> tuple[int, int]:
        """Select MORI kernel launch config based on batch size."""
        if num_tokens > _PREFILL_TOKEN_THRESHOLD:
            return _PREFILL_BLOCK_NUM, _PREFILL_WARP_PER_BLOCK
        return _DECODE_BLOCK_NUM, _DECODE_WARP_PER_BLOCK

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def output_is_reduced(self) -> bool:
        return True

    def num_dispatchers(self):
        return self.num_dispatchers_

    def max_num_tokens_per_rank(self) -> int | None:
        return self.max_tokens_per_rank

    def topk_indices_dtype(self) -> torch.dtype | None:
        return torch.int32

    def supports_async(self) -> bool:
        return False

    def post_init_setup(self, fused_experts: mk.FusedMoEExperts):
        super().post_init_setup(fused_experts)
        if self.use_fp8_dispatch:
            from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (
                AiterExperts,
            )

            if isinstance(fused_experts, AiterExperts):
                fused_experts._accepts_prequantized_fp8 = True
                logger.info_once(
                    "AITER FP8 dispatch enabled: tokens dispatched as FP8 "
                    "over MORI RDMA (2x bandwidth savings vs BF16)."
                )

    def prepare(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool = False,
    ) -> mk.PrepareResultType:
        """
        Returns a tuple of:
        - quantized + dispatched a.
        - Optional quantized + dispatched a1_scales.
        - Optional ExpertTokensMetadata containing gpu/cpu tensors
          as big as the number of local experts with the information about the
          number of tokens assigned to each local expert.
        - Optional dispatched expert topk IDs
        - Optional dispatched expert topk weight
        """
        assert not apply_router_weight_on_input, (
            "mori does not support apply_router_weight_on_input=True now."
        )
        scale = None
        if defer_input_quant:
            # Expert kernel (e.g. AITER) quantizes inputs internally.
            # Dispatch raw BF16 data; 2x RDMA bandwidth vs FP8 dispatch.
            pass
        elif self.use_fp8_dispatch:
            from aiter import QuantType, get_hip_quant

            if quant_config.is_block_quantized:
                quant_func = get_hip_quant(QuantType.per_1x128)
                a1, scale = quant_func(a1, quant_dtype=current_platform.fp8_dtype())
            elif quant_config.is_per_act_token:
                quant_func = get_hip_quant(QuantType.per_Token)
                a1, scale = quant_func(a1, quant_dtype=current_platform.fp8_dtype())

        block_num, warp_per_block = self._launch_config(a1.shape[0])
        (
            dispatch_a1,
            dispatch_weights,
            dispatch_scale,
            dispatch_ids,
            dispatch_recv_token_num,
        ) = self.mori_op.dispatch(
            a1, topk_weights, scale, topk_ids, block_num, warp_per_block
        )

        expert_tokens_meta = mk.ExpertTokensMetadata(
            expert_num_tokens=dispatch_recv_token_num, expert_num_tokens_cpu=None
        )

        return (
            dispatch_a1,
            dispatch_scale,
            expert_tokens_meta,
            dispatch_ids,
            dispatch_weights,
        )

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> None:
        num_token = output.shape[0]
        block_num, warp_per_block = self._launch_config(num_token)
        result = self.mori_op.combine(
            fused_expert_output,
            None,
            topk_ids,
            block_num,
            warp_per_block,
        )[0]
        output.copy_(result[:num_token])
