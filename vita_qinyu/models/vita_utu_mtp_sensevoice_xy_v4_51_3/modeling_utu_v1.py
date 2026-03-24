import math
from functools import partial
from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    can_return_tuple,
    is_torch_flex_attn_available,
    logging,
    replace_return_docstrings,
    is_flash_attn_2_available,
)
# for >= 4.51.3 compatibility
import transformers
from packaging.version import Version
if Version(transformers.__version__) <= Version("4.52.3"):
    from transformers.utils import LossKwargs
else:
    from transformers.utils import TransformersKwargs as LossKwargs

from transformers.utils.deprecation import deprecate_kwarg
from .configuration_utu_v1 import UTUV1Config

try:
    from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss
    from liger_kernel.transformers.rms_norm import LigerRMSNorm
    from liger_kernel.transformers.rope import liger_rotary_pos_emb
    from liger_kernel.transformers.swiglu import LigerSwiGLUMLP
    from liger_kernel.ops.rope import LigerRopeFunction
except Exception as e:
    print(e)

if is_torch_flex_attn_available():
    from torch.nn.attention.flex_attention import BlockMask

    from transformers.integrations.flex_attention import make_flex_block_causal_mask

is_aiter_available = False

if is_flash_attn_2_available():
    try:
        from aiter import flash_attn_varlen_func
        is_aiter_available = True
    except ImportError:
        from flash_attn import flash_attn_varlen_func
else:
    flash_attn_varlen_func = None

logger = logging.get_logger(__name__)
_CONFIG_FOR_DOC = "UTUV1Config"

from .modeling_sensevoice import SenseVoice
from .resampler_projector import VisionResamplerProjector
from .resampler_projector import AudioResamplerProjector

from .speaker_projector import SpeakerProjector
text_vocab_size = None

def liger_rotary_pos_emb_interleave(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """
    Applies Rotary Positional Embedding (RoPE) operation to query and key states.

    Args:
        q (torch.Tensor): The query tensor of shape (bsz, n_q_head, seq_len, head_dim).
        k (torch.Tensor): The key tensor of shape (bsz, n_kv_head, seq_len, head_dim).
        cos (torch.Tensor): The cosine tensor of shape (1, seq_len, head_dim) or (bsz, seq_len, head_dim).
        sin (torch.Tensor): The sine tensor of shape (1, seq_len, head_dim) or (bsz, seq_len, head_dim).
        position_ids (torch.Tensor, optional): The position ids tensor. Defaults to None.
        unsqueeze_dim (int, optional): The dimension to unsqueeze. Defaults to 1.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The query and key tensors after applying the RoPE operation.
    """

    b, h, s, d = q.shape
    q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    b, h, s, d = k.shape
    k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    return LigerRopeFunction.apply(q, k, cos, sin, position_ids, unsqueeze_dim)


def fixed_cross_entropy(shift_hidden_states, shift_labels, lm_head_weights, num_items_in_batch=None, ignore_index=-100, **kwargs):
    reduction = "sum" if num_items_in_batch is not None else "mean"
    lce = LigerFusedLinearCrossEntropyLoss(reduction=reduction, ignore_index=ignore_index)
    loss = lce(lm_head_weights, shift_hidden_states, shift_labels)
    if reduction == "sum":
        loss = loss / num_items_in_batch
    return loss


def ForCausalLMLoss(
    hidden_states, labels, lm_head_weights, hidden_size, vocab_size, num_items_in_batch=None, ignore_index=-100, shift=True, **kwargs
):
    if shift:
        shift_hidden_states = hidden_states[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
    else: # already shifted
        shift_hidden_states = hidden_states.contiguous()
        shift_labels = labels.contiguous()
    # flatten tokens
    shift_hidden_states = shift_hidden_states.view(-1, hidden_size)
    shift_labels = shift_labels.view(-1)

    loss = fixed_cross_entropy(shift_hidden_states=shift_hidden_states, shift_labels=shift_labels, lm_head_weights=lm_head_weights,
                               num_items_in_batch=num_items_in_batch, ignore_index=ignore_index, **kwargs)
    return loss


class UTUV1RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        UTUV1RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class UTUV1RotaryEmbedding(nn.Module):
    def __init__(self, config: UTUV1Config, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class UTUV1MLP(nn.Module):
    def __init__(self, config, hidden_size=None, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class UTUV1TopkRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob

        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, config.hidden_size)))
        self.register_buffer("e_score_correction_bias", torch.zeros((self.n_routed_experts)))

    @torch.no_grad()
    def get_topk_indices(self, scores):
        scores_for_choice = scores.view(-1, self.n_routed_experts) + self.e_score_correction_bias.unsqueeze(0)
        group_scores = (
            scores_for_choice.view(-1, self.n_group, self.n_routed_experts // self.n_group)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.n_group, self.n_routed_experts // self.n_group)
            .reshape(-1, self.n_routed_experts)
        )
        scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        return topk_indices

    def forward(self, hidden_states):
        hidden_states = hidden_states.view(-1, self.config.hidden_size)
        router_logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32))
        scores = router_logits.sigmoid()
        topk_indices = self.get_topk_indices(scores)
        topk_weights = scores.gather(1, topk_indices)
        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights /= denominator
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_indices, topk_weights


class UTUV1MoE(nn.Module):
    """
    A mixed expert module containing shared experts.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gate = UTUV1TopkRouter(config)
        if self.training:
            self.experts = nn.ModuleList(
                [
                    LigerSwiGLUMLP(config, intermediate_size=config.moe_intermediate_size)
                    for _ in range(config.n_routed_experts)
                ]
            )
            self.shared_experts = LigerSwiGLUMLP(
                config=config, intermediate_size=config.moe_intermediate_size * config.n_shared_experts
            )
        else:
            self.experts = nn.ModuleList(
                [
                    UTUV1MLP(config, intermediate_size=config.moe_intermediate_size)
                    for _ in range(config.n_routed_experts)
                ]
            )
            self.shared_experts = UTUV1MLP(
                config=config, intermediate_size=config.moe_intermediate_size * config.n_shared_experts
            )

    def moe(self, hidden_states: torch.Tensor, topk_indices: torch.Tensor, topk_weights: torch.Tensor):
        r"""
        CALL FOR CONTRIBUTION! I don't have time to optimise this right now, but expert weights need to be fused
        to not have to do a loop here (utu has 256 experts soooo yeah).
        """
        final_hidden_states = torch.zeros_like(hidden_states, dtype=topk_weights.dtype)
        expert_mask = torch.nn.functional.one_hot(topk_indices, num_classes=len(self.experts))
        expert_mask = expert_mask.permute(2, 0, 1)

        for expert_idx in range(len(self.experts)):
            expert = self.experts[expert_idx]
            mask = expert_mask[expert_idx]
            token_indices, weight_indices = torch.where(mask)

            if token_indices.numel() > 0:
                expert_weights = topk_weights[token_indices, weight_indices]
                expert_input = hidden_states[token_indices]
                expert_output = expert(expert_input)
                weighted_output = expert_output * expert_weights.unsqueeze(-1)
                final_hidden_states.index_add_(0, token_indices, weighted_output)

        # in original utu, the output of the experts are gathered once we leave this module
        # thus the moe module is itelsf an IsolatedParallel module
        # and all expert are "local" meaning we shard but we don't gather
        return final_hidden_states.type(hidden_states.dtype)

    def forward(self, hidden_states):
        residuals = hidden_states
        orig_shape = hidden_states.shape
        topk_indices, topk_weights = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        hidden_states = self.moe(hidden_states, topk_indices, topk_weights).view(*orig_shape)
        hidden_states = hidden_states + self.shared_experts(residuals)
        return hidden_states


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


def apply_rotary_pos_emb_interleave(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    r"""
    TODO let's just use the original freqcis computation to not have the view
    transpose + reshape! This is not optimized!
    Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    b, h, s, d = q.shape
    q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    b, h, s, d = k.shape
    k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


try:
    import liger_kernel
except Exception:
    LigerRMSNorm = UTUV1RMSNorm
    liger_rotary_pos_emb = apply_rotary_pos_emb
    LigerSwiGLUMLP = UTUV1MLP


def yarn_get_mscale(scale=1, mscale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


class UTUV1Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: UTUV1Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.attention_dropout = config.attention_dropout
        self.num_heads = config.num_attention_heads
        self.rope_theta = config.rope_theta
        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_head_dim = config.qk_head_dim

        self.is_causal = True

        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.qk_head_dim, bias=False)
        else:
            self.q_a_proj = nn.Linear(config.hidden_size, config.q_lora_rank, bias=config.attention_bias)
            if self.training:
                self.q_a_layernorm = LigerRMSNorm(config.q_lora_rank)
            else:
                self.q_a_layernorm = UTUV1RMSNorm(config.q_lora_rank)
            self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False)

        self.kv_a_proj_with_mqa = nn.Linear(
            config.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=config.attention_bias,
        )
        if self.training:
            self.kv_a_layernorm = LigerRMSNorm(self.kv_lora_rank)
        else:
            self.kv_a_layernorm = UTUV1RMSNorm(self.kv_lora_rank)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )

        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )

        self.scaling = self.qk_head_dim ** (-0.5)
        if self.config.rope_scaling is not None:
            mscale_all_dim = self.config.rope_scaling.get("mscale_all_dim", 0)
            scaling_factor = self.config.rope_scaling["factor"]
            if mscale_all_dim:
                mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
                self.scaling = self.scaling * mscale * mscale

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        batch_size, seq_length = hidden_states.shape[:-1]
        query_shape = (batch_size, seq_length, -1, self.qk_head_dim)
        key_shape = (batch_size, seq_length, -1, self.qk_nope_head_dim + self.v_head_dim)

        if self.q_lora_rank is None:
            q_states = self.q_proj(hidden_states).view(query_shape).transpose(1, 2)
        else:
            q_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states))).view(query_shape).transpose(1, 2)
        q_pass, q_rot = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        k_pass, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

        k_pass = self.kv_b_proj(self.kv_a_layernorm(k_pass)).view(key_shape).transpose(1, 2)
        k_pass, value_states = torch.split(k_pass, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)

        cos, sin = position_embeddings
        if self.config.rope_interleave:  # support using interleaved weights for efficiency
            if self.training:
                q_rot, k_rot = liger_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin)
            else:
                q_rot, k_rot = apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin)
        else:
            if self.training:
                q_rot, k_rot = liger_rotary_pos_emb(q_rot, k_rot, cos, sin)
            else:
                q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin)
        k_rot = k_rot.expand(*k_pass.shape[:-1], -1)

        query_states = torch.cat((q_pass, q_rot), dim=-1)
        key_states = torch.cat((k_pass, k_rot), dim=-1)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        if self.config._attn_implementation == "flash_attention_2" and self.qk_head_dim != self.v_head_dim:
            value_states = F.pad(value_states, [0, self.qk_head_dim - self.v_head_dim])

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        # print(f"{kwargs.keys()=}")
        if "cu_seq_lens" not in kwargs:
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                **kwargs,
            )
            if self.config._attn_implementation == "flash_attention_2" and self.qk_head_dim != self.v_head_dim:
                attn_output = attn_output[:, :, :, : self.v_head_dim]
        else:
            cu_seq_lens = kwargs["cu_seq_lens"].to(torch.int32)
            max_seq_len = kwargs["max_seq_len"]

            query_states = query_states.squeeze(0).transpose(0, 1)
            key_states = key_states.squeeze(0).transpose(0, 1)
            value_states = value_states.squeeze(0).transpose(0, 1)

            if is_aiter_available:
                attn_output = flash_attn_varlen_func(query_states, key_states, value_states, cu_seq_lens, cu_seq_lens, max_seq_len, max_seq_len, \
                    dropout_p=0.0 if not self.training else self.attention_dropout,
                    softmax_scale=self.scaling,
                    causal=self.is_causal, return_lse=True)[0]
            else:
                attn_output = flash_attn_varlen_func(query_states, key_states, value_states, cu_seq_lens, cu_seq_lens, max_seq_len, max_seq_len, \
                    dropout_p=0.0 if not self.training else self.attention_dropout,
                    softmax_scale=self.scaling,
                    causal=self.is_causal)

            attn_output = attn_output.unsqueeze(0)
            attn_output = attn_output[:, :, :, : self.v_head_dim]
            attn_weights = None

        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class UTUV1DecoderLayer(nn.Module):
    def __init__(self, config: UTUV1Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = UTUV1Attention(config=config, layer_idx=layer_idx)

        if layer_idx >= config.first_k_dense_replace:
            self.mlp = UTUV1MoE(config)
        else:
            if self.training:
                self.mlp = LigerSwiGLUMLP(config)
            else:
                self.mlp = UTUV1MLP(config)

        if self.training:
            self.input_layernorm = LigerRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.post_attention_layernorm = LigerRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.input_layernorm = UTUV1RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.post_attention_layernorm = UTUV1RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


UTU_V1_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`UTUV1Config`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare UTUV1 Model outputting raw hidden-states without any specific head on top.",
    UTU_V1_START_DOCSTRING,
)
class UTUV1PreTrainedModel(PreTrainedModel):
    config_class = UTUV1Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["UTUV1DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_attention_backend = True

    def init_weights(self):
        """
        If needed prunes and maybe initializes weights. If using a custom `PreTrainedModel`, you need to implement any
        initialization logic in `_init_weights`.
        """
        # Prune heads if needed
        if self.config.pruned_heads:
            self.prune_heads(self.config.pruned_heads)

        if "-init" in self.name_or_path:
            # Initialize weights
            self.apply(self._initialize_weights)

            # Adjust weights of o_proj in Attention and down_proj in MLP
            for name, module in self.named_modules():
                if "o_proj" in name or "down_proj" in name:
                    # For the output projection, we reinitialize the weights
                    scaled_std = self.config.initializer_range * (1.0 / self.config.num_hidden_layers) ** 0.5
                    module.weight.data.normal_(mean=0.0, std=scaled_std)

            # Tie weights should be skipped when not initializing all weights
            # since from_pretrained(...) calls tie weights anyways
            self.tie_weights()

    def _init_weights(self, module):
        std = self.config.initializer_range
        embedding_std = self.config.embedding_initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=embedding_std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, UTUV1TopkRouter):
            module.weight.data.normal_(mean=0.0, std=std)
        elif isinstance(module, nn.Parameter):
            module.weight.data.normal_(mean=0.0, std=std)
        elif isinstance(module, UTUV1RMSNorm) or isinstance(module, LigerRMSNorm):
            module.weight.data.fill_(1.0)


UTU_V1_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
"""


@add_start_docstrings(
    "The bare UTUV1 Model outputting raw hidden-states without any specific head on top.",
    UTU_V1_START_DOCSTRING,
)
class UTUV1Model(UTUV1PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`UTUV1DecoderLayer`]

    Args:
        config: UTUV1Config
    """

    _keys_to_ignore_on_load_unexpected = [r"model\.layers\.61.*"]

    def __init__(self, config: UTUV1Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [UTUV1DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        if self.training:
            self.norm = LigerRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = UTUV1RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = UTUV1RotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        self.audio_model = SenseVoice(config)
        self.audio_projection = AudioResamplerProjector(512, config.hidden_size)

        if config.use_speaker_embedding:
            self.speaker_projection = SpeakerProjector(
                config.speaker_embedding_size,
                config.hidden_size,
                getattr(config, 'audio_projection_norm_type', 'layer_norm')
            )

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @can_return_tuple
    @add_start_docstrings_to_model_forward(UTU_V1_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        images: Optional[torch.FloatTensor] = None,
        image_indices: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        audios: Optional[torch.FloatTensor] = None,
        audio_indices: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        layer_idxs = None,
        audio_tokens : Optional[torch.LongTensor] = None,
        audio_token_starts : Optional[torch.LongTensor] = None,
        audio_token_ends : Optional[torch.LongTensor] = None,
        speaker_embeddings : Optional[torch.FloatTensor] = None,
        speaker_mask : Optional[torch.Tensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutputWithPast:

        if (past_key_values is None or len(past_key_values) == 0) and speaker_embeddings is not None and speaker_embeddings.numel() > 0:
            fake_speakers = None
            assert len(speaker_embeddings) == 1, speaker_embeddings.shape
            speaker_embeds = self.speaker_projection(speaker_embeddings)[0] # N x H
            assert speaker_embeds.shape[0] == len(speaker_embeddings[0]), f'{speaker_embeds.shape[0]} == {len(speaker_embeddings[0])} {speaker_embeds.shape}, {speaker_embeddings.shape}'
        elif self.training and hasattr(self, "speaker_projection"):
            device = self.get_input_embeddings().weight.data.device
            dtype = self.get_input_embeddings().weight.data.dtype
            fake_speakers = torch.ones((1, 1, 192), dtype=dtype, device=device)
            speaker_embeds = self.speaker_projection(fake_speakers)
        else:
            fake_speakers = None
            speaker_embeds = None

        if (past_key_values is None or len(past_key_values) == 0) and audios is not None:
            audio_embeds, audio_lengths = self.audio_model(audios)
            # if torch.distributed.get_rank() == 0:
            #     print(f"audio_embeds {audio_embeds.size()}")
            assert audio_embeds.shape[0] == len(audios)
            fake_audios = None
            audio_embeds = self.audio_projection(audio_embeds)

        elif self.training:
            device = self.get_input_embeddings().weight.data.device
            dtype = self.get_input_embeddings().weight.data.dtype
            fake_audios = torch.ones((1, 1, 560), dtype=dtype, device=device)
            audio_embeds, audio_lengths = self.audio_model(fake_audios)
            audio_embeds = self.audio_projection(audio_embeds)
        else:
            fake_audios = None
            audio_embeds = None


        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        # TODO (joao): remove this exception in v4.56 -- it exists for users that try to pass a legacy cache
        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        assert not torch.isnan(inputs_embeds).any(), f"audio embeds {inputs_embeds=} {inputs_embeds.size()=}"

        if fake_audios is not None:
            inputs_embeds = inputs_embeds + audio_embeds.mean() * 0.0
        elif audio_embeds is not None:
            inputs_embeds = inputs_embeds.clone()
            for audio_embeds_, audio_lengths_, audio_indices_ in zip(audio_embeds, audio_lengths, audio_indices,):
                # print(f"{audio_embeds_.size()=} {audio_lengths_=} {audio_indices_.size()=}")
                audio_embeds_ = audio_embeds_[:audio_lengths_, ...]
                audio_embeds_ = audio_embeds_.to(inputs_embeds.device)
                indices_b, indices_s = audio_indices_.to(inputs_embeds.device).unbind(dim=0)
                inputs_embeds[indices_b.view(-1), indices_s.view(-1)] = audio_embeds_.view(-1, audio_embeds_.shape[-1])
        assert not torch.isnan(inputs_embeds).any(), f"audio embeds {inputs_embeds=} {inputs_embeds.size()=}"
            # inputs_embeds = inputs_embeds + audio_embeds.mean() * 0.0

        if fake_speakers is not None:
            inputs_embeds = inputs_embeds + speaker_embeds.mean() * 0.0
        elif speaker_embeds is not None:
            # inputs_embeds = inputs_embeds.clone()
            inputs_embeds[speaker_mask.bool()] = speaker_embeds
        assert not torch.isnan(inputs_embeds).any(), f"audio embeds {inputs_embeds=} {inputs_embeds.size()=}"

        if audio_tokens is not None and audio_tokens.numel() > 0:
            audio_tokens_embeds = self.embed_tokens(audio_tokens) # 8 x T => 8 x T x H
            audio_start_positions, audio_end_positions = [], []
            audio_token_mask = torch.zeros(inputs_embeds.shape[:-1]).bool() # B x T
            if len(audio_token_starts) == 1:
                for _audio_start, _audio_end in zip(audio_token_starts[0], audio_token_ends[0]):
                    audio_token_mask[:, _audio_start: _audio_end] = True
            else:
                # audio_token_starts B x T, audio_token_ends B x T
                for i, (_audio_token_starts, _audio_token_ends) in enumerate(zip(audio_token_starts, audio_token_ends)):
                    for _audio_start, _audio_end in zip(_audio_token_starts, _audio_token_ends):
                        audio_token_mask[i, _audio_start: _audio_end] = True
                        import pdb; pdb.set_trace()
            inputs_embeds[audio_token_mask] = audio_tokens_embeds.squeeze(0).mean(0) # 8 x T x H => T x H
            assert not torch.isnan(inputs_embeds).any(), f"{inputs_embeds=} {inputs_embeds.size()=}"
        assert not torch.isnan(self.embed_tokens.weight).any(), f"{self.embed_tokens.weight=} {self.embed_tokens.weight.size()=}"

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    partial(decoder_layer.__call__, **flash_attn_kwargs),
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **flash_attn_kwargs,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool = False,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and (attention_mask == 0.0).any():
                return attention_mask
            return None
        if self.config._attn_implementation == "flex_attention":
            if isinstance(attention_mask, torch.Tensor):
                attention_mask = make_flex_block_causal_mask(attention_mask)
            if isinstance(attention_mask, BlockMask):
                return attention_mask

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type in ["cuda", "xpu"]
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            min_dtype = torch.finfo(dtype).min
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        **kwargs,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape
                `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache,
                to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            device (`torch.device`):
                The device to place the 4D attention mask on.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                    causal_mask.device
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )

        return causal_mask


class KwargsForCausalLM(FlashAttentionKwargs, LossKwargs): ...


class VITAUTUV1ForCausalLM(UTUV1PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = UTUV1Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

#          if config.use_audio_heads:
            #  self.audio_heads = nn.ModuleList([
                #  nn.Linear(config.hidden_size, config.codebook_size, bias=False)
                #  for _ in range(config.num_codebook-1)
#              ])
        if config.use_audio_heads:
            # 原有的独立heads
            self.audio_heads = nn.ModuleList([
                nn.Linear(config.hidden_size, config.codebook_size, bias=False)
                for _ in range(config.num_codebook-1)
            ])

            # 新增: 融合层
            total_audio_vocab = config.codebook_size * (config.num_codebook - 1)
            self.fused_audio_head = nn.Linear(
                config.hidden_size,
                total_audio_vocab,  # 例如: 7 * 1024 = 7168
                bias=False
            )

            # 复制权重: 把7个head的权重拼接
            with torch.no_grad():
                fused_weight = torch.cat([head.weight for head in self.audio_heads], dim=0)
                self.fused_audio_head.weight.copy_(fused_weight)


        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @can_return_tuple
    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
    @add_start_docstrings_to_model_forward(UTU_V1_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        images: Optional[torch.FloatTensor] = None,
        image_indices: Optional[torch.LongTensor] = None,
        audios: Optional[torch.FloatTensor] = None,
        audio_indices: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        audio_feature_lengths: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        video_second_per_grid: Optional[torch.LongTensor] = None,
        use_audio_in_video: Optional[bool] = None,
        processor=None,
        audio_tokens : Optional[torch.LongTensor] = None,
        audio_labels : Optional[torch.LongTensor] = None,
        audio_token_starts : Optional[torch.LongTensor] = None,
        audio_token_ends : Optional[torch.LongTensor] = None,
        first_audio_token_id: Optional[torch.LongTensor] = None,
        speaker_embeddings : Optional[torch.FloatTensor] = None,
        speaker_mask : Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> CausalLMOutputWithPast:
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            logits_to_keep (`int` or `torch.Tensor`, *optional*):
                If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
                If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
                This is useful when using packed tensor format (single dimension for batch and sequence length).

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, UTUV1ForCausalLM

        >>> model = UTUV1ForCausalLM.from_pretrained("meta-utu_v1/UTUV1-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-utu_v1/UTUV1-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        global text_vocab_size
        text_vocab_size = first_audio_token_id

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=images,
            image_indices=image_indices,
            image_grid_thw=image_grid_thw,
            audios=audios,
            audio_indices=audio_indices,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            audio_tokens=audio_tokens,
            audio_token_starts=audio_token_starts,
            audio_token_ends=audio_token_ends,
            speaker_embeddings=speaker_embeddings,
            speaker_mask=speaker_mask,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        # import pdb; pdb.set_trace
        if self.training:
            logits = None
#          else:
            #  logits = self.lm_head(hidden_states[:, slice_indices, :])
            #  if text_vocab_size is not None and hasattr(self, 'audio_heads'):
                #  for i, audio_head in enumerate(self.audio_heads):
                    #  audio_logits = audio_head(hidden_states[:, slice_indices, :])
                    #  audio_logits_s = text_vocab_size + i * codebook_size
                    #  audio_logits_e = text_vocab_size + (i+1) * codebook_size
#                      logits[:,:,audio_logits_s:audio_logits_e] = audio_logits
        else:
            logits = self.lm_head(hidden_states[:, slice_indices, :])

            if text_vocab_size is not None and hasattr(self, 'fused_audio_head'):
                # 一次计算所有audio logits (无循环!)
                all_audio_logits = self.fused_audio_head(hidden_states[:, slice_indices, :])
                # 直接赋值
                audio_vocab_size = all_audio_logits.shape[-1]
                logits[:, :, text_vocab_size:text_vocab_size+audio_vocab_size] = all_audio_logits



        # |   T label   | A label   |  T label    | A label   | T label |
        # |t1||t2|t3|BOA|a1|a2|a3|a5|EOA|t5|t6|BOA|a6|a7|a8|a9|EOA|t7|t8|
        # | T token |  A token   |  T token   |  A token   |  T token   |
        # Therefore shift Text logits & labels in ForCausalLMLoss
        # do not shift Audio logits & labels in ForCausalLMLoss
        loss = None
        if labels is not None:
            ce_loss_text, ce_loss_audio, loss = self.compute_xy_loss(
                hidden_states[:, slice_indices, :], labels, audio_labels,
                audio_token_starts,
                audio_token_ends,
                mtp_idx=-1,
                **kwargs
            )
        # loss = self.compute_xy_loss(hidden_states[:, slice_indices, :], labels, audio_labels,audio_token_starts, audio_token_ends, mtp_idx=-1,**kwargs)
        # assert not torch.isnan(logits).any(), f"{logits=} {logits.size()=}"

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def compute_xy_loss(
        self,
        hidden_states, labels, audio_labels,
        audio_token_starts,
        audio_token_ends,
        mtp_idx=-1,
        **kwargs
    ):
        global text_vocab_size

        num_codebook = self.config.num_codebook
        codebook_size = self.config.codebook_size
        audio_token_mask = torch.zeros(hidden_states.shape[:-1]).bool()
        audio_label_mask = torch.zeros(hidden_states.shape[:-1]).bool()
        for _audio_start, _audio_end in zip(audio_token_starts[0], audio_token_ends[0]):
            audio_label_mask[:, _audio_start: _audio_end] = True
            audio_token_mask[:, _audio_start-1: _audio_end-1] = True
        text_token_mask = ~audio_token_mask
        text_label_mask = ~audio_label_mask
        # B x T x H
        audio_vocab_size = (num_codebook-1) * codebook_size
        # logits_text = torch.cat([
        #     logits[..., :text_vocab_size].contiguous(),
        #     # logits.new(*logits.shape[:-1], (num_codebook-1)*codebook_size).fill_(torch.finfo(logits.dtype).min),
        #     logits[..., text_vocab_size+audio_vocab_size:].contiguous(),
        # ], dim=-1)
        # # import pdb; pdb.set_trace()

        # skip 7 codebook (7 * 1032)
        labels_text = labels.contiguous()
        labels_text[(labels_text>=0)&(labels_text>=text_vocab_size+audio_vocab_size)] -= audio_vocab_size
        lm_head_weights_no_audio = torch.cat([
            self.lm_head.weight[:text_vocab_size],
            self.lm_head.weight[text_vocab_size+audio_vocab_size:]
        ], dim=0)

        ce_loss_text = ForCausalLMLoss(
            hidden_states=hidden_states,
            lm_head_weights=lm_head_weights_no_audio,
            labels=labels_text,
            vocab_size=lm_head_weights_no_audio.shape[0], # V x H
            hidden_size=self.config.hidden_size,
            shift=True,
            **kwargs
        )


        num_text_labels = labels[text_label_mask].ne(-100).sum()
        num_audio_labels = audio_token_mask.sum()
        num_labels = labels.ne(-100).sum()
        # import pdb; pdb.set_trace()
        # assert (labels[audio_label_mask] >= 151691).all()
        # if is_main_process():
        #     logger.info(
        #         f"{num_text_labels=} + {num_audio_labels=} == {num_labels=} {logits.shape=} {labels.shape=} {audio_label_mask.sum()=}")
        assert num_text_labels + num_audio_labels == num_labels, \
            f"{mtp_idx=} {num_text_labels=} + {num_audio_labels=} == {num_labels=}"

        ce_loss_audio, ce_loss_audios = 0, []

        if audio_label_mask.any():
            for i in range(1, num_codebook): # text_vocab_size start from <audio_1_0>
                logits_s = text_vocab_size + (i-1)*codebook_size
                logits_e = text_vocab_size + i*codebook_size
                if hasattr(self, 'audio_heads'):
                    lm_head_weights_audio_i = self.audio_heads[i-1].weight
                else:
                    lm_head_weights_audio_i = self.lm_head.weight[logits_s:logits_e]
                labels_audio_i = audio_labels.squeeze(0)[i]

                ce_loss_audio_i = ForCausalLMLoss(
                    hidden_states=hidden_states[audio_token_mask],
                    lm_head_weights=lm_head_weights_audio_i,
                    labels=labels_audio_i,
                    vocab_size=codebook_size,
                    hidden_size=self.config.hidden_size,
                    shift=False,
                    **kwargs
                )
                ce_loss_audios.append(ce_loss_audio_i)
            ce_loss_audio = torch.stack(ce_loss_audios).mean()
            assert not torch.isnan(ce_loss_audio).any(), f"{ce_loss_audio=} {ce_loss_audio.size()=}"
        losses = [ce_loss_text] + ce_loss_audios
        _losses = torch.stack(losses).tolist()

        loss = ce_loss_text + ce_loss_audio
        if is_main_process():
            logger.warning(f"mtp_idx {mtp_idx}\tavg: {loss} | text: {ce_loss_text} | audio: {ce_loss_audio} | all losses: {_losses}")
        assert not torch.isnan(loss).any(), f"{loss=} {loss.size()=}"
        return ce_loss_text, ce_loss_audio, loss


def is_main_process(local=True):
    import os
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
        if local:
            rank = int(os.environ["LOCAL_RANK"])
        else:
            rank = torch.distributed.get_rank()
        _is_main_process = rank == 0
        return _is_main_process
    return True



__all__ = ["UTUV1PreTrainedModel", "UTUV1Model", "VITAUTUV1ForCausalLM"]
