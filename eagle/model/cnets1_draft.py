# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
""" PyTorch LLaMA model."""
import copy
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import math
import threading
from typing import List, Optional, Tuple, Union
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
import numpy as np
from transformers.activations import ACT2FN
from huggingface_hub import hf_hub_download
from transformers import AutoConfig
from accelerate import init_empty_weights
import torch.nn as nn
import time
from thop import profile
try:
    from .configs import EConfig
    from .utils_c import *
    from .choices import *
except:
    from configs import EConfig
    from utils_c import *
    from choices import *
    from utils import prepare_logits_processor
from collections import defaultdict
# from .device_manager import DEVICE_DRAFT
from .config_loader import config as StarSD_config


def load_embeddingLayer(model_dir, weight_dir, device):
    tryconfig = AutoConfig.from_pretrained(model_dir)

    embedding_weights = torch.load(weight_dir, map_location="cpu")

    new_weight = torch.nn.Parameter(
        embedding_weights["weight"].to(torch.float16).to("cpu")  # Changed to float16, consistent with cnets1
    )

    tryembed = nn.Embedding(
        tryconfig.vocab_size, 
        tryconfig.hidden_size,
        _weight=new_weight,
        dtype=torch.float16  # Changed to float16, consistent with cnets1
    ).to("cpu")

    print("Load Embedding Layer Successfully")
    print("dtype of weight:", tryembed.weight.dtype) 
    # print("\n\nAnother Embedd:\n",tryembed.weight.data)
    return tryembed.weight.data.to(device)


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
        input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


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


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)


class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                    (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        if hasattr(config, "qkv_bias"):
            self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.qkv_bias)  # 移除dtype，与cnets1一致
            self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.qkv_bias)  # 移除dtype
            self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.qkv_bias)  # 移除dtype
        else:
            self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)  # 移除dtype
            self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)  # 移除dtype
            self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)  # 移除dtype
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)  # 移除dtype
        
        print(f"[LlamaAttention Init] q_proj.weight.dtype: {self.q_proj.weight.dtype}")
        print(f"[LlamaAttention Init] k_proj.weight.dtype: {self.k_proj.weight.dtype}")
        print(f"[LlamaAttention Init] v_proj.weight.dtype: {self.v_proj.weight.dtype}")
        print(f"[LlamaAttention Init] o_proj.weight.dtype: {self.o_proj.weight.dtype}")
        
        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            if hasattr(self.config, "rope_theta"):
                self.rotary_emb = LlamaRotaryEmbedding(self.head_dim,
                                                       max_position_embeddings=self.max_position_embeddings,
                                                       base=self.config.rope_theta)
            else:
                self.rotary_emb = LlamaRotaryEmbedding(self.head_dim,
                                                       max_position_embeddings=self.max_position_embeddings)
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim, max_position_embeddings=self.max_position_embeddings, scaling_factor=scaling_factor
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim, max_position_embeddings=self.max_position_embeddings, scaling_factor=scaling_factor
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            # print("query_states:   ",query_states.dtype, query_states.device)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

        print(f"[LlamaMLP Init] gate_proj.weight.dtype: {self.gate_proj.weight.dtype}")
        print(f"[LlamaMLP Init] up_proj.weight.dtype: {self.up_proj.weight.dtype}")
        print(f"[LlamaMLP Init] down_proj.weight.dtype: {self.down_proj.weight.dtype}")

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj



class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
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


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config, index):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config=config)
        self.mlp = LlamaMLP(config)
        self.index = index
        if self.index != 0:
            self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        if self.index != 0:
            hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
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

        if use_cache:
            outputs += (present_key_value,)

        return outputs




class I(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.ones(1, dtype=torch.float32))

    def forward(self, x):
        return x + self.dummy - self.dummy  # (also tried x+self.dummy)


def len_list(x, n):
    return [i for i in x if len(i) <= n]



def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


class ModelDynamic(nn.Module):
    def __init__(self, config, load_emb=True, path=None, bias=True, total_tokens=63, threshold=1.0, ea_model_path=None):
        super().__init__()

        self.gradient_checkpointing = True
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        if load_emb:
            self.embed_tokens.weight.data = load_embeddingLayer(model_dir="lmsys/vicuna-7b-v1.3", 
                                                              weight_dir=os.path.join(ea_model_path, "embeddingLayer/vicuna7b_embeddings.bin"), 
                                                              device=StarSD_config.device_draft)

        # Remove fixed depth and top_k initialization
        self.total_tokens = total_tokens - 1
        self.threshold = math.log(threshold)
        
        # Multi-client state management
        self.client_states = {}  # Store state for each client
        self._lock = threading.RLock()  # Thread-safe lock
        self.inference_time = ()
        self.prepare_time = 0.0
        self.construction_time = 0.0
        self.inference_time_per_depth = []  # 🔥 Inference time list for each depth
        self.time_list = []

        # self.layers = nn.ModuleList([LlamaDecoderLayer(config, index) for index in range(config.num_hidden_layers)])
        self.layers = nn.ModuleList([
            self._build_decoder_layer(config, idx) 
            for idx in range(config.num_hidden_layers)
        ])
        print("EMBED DTYPE: ", self.embed_tokens.weight.dtype)
        self.fc = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=bias, dtype=torch.float16)  # Changed to float16, consistent with cnets1
        self.act = ACT2FN[config.hidden_act]
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        
        
        for param in self.embed_tokens.parameters():
            param.requires_grad = False
        



    def _build_decoder_layer(self, config, idx):
        layer = LlamaDecoderLayer(config, idx)
        layer = layer.to(torch.float16)
        return layer


    def init_tree(self, top_k):
        """Initialize tree structures with dynamic top_k"""
        self.tree_mask_init = torch.eye(top_k, device=self.embed_tokens.weight.device)[None, None]
        self.position_ids = torch.zeros(top_k, device=self.embed_tokens.weight.device, dtype=torch.long)
        self.tree_mask_init = self.tree_mask_init.to(self.embed_tokens.weight.device)

    def reset(self):
        self.tree_mask = None
    

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                torch.float32,  # force to cast to float32
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            expanded_attn_mask = _expand_mask(attention_mask, torch.float32, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        if hasattr(self, "tree_mask") and self.tree_mask is not None:
            tree_mask = self.tree_mask
            _, _, tree_shape0, tree_shape1 = tree_mask.shape
            combined_attention_mask[:, :, -tree_shape0:, -tree_shape1:][
                tree_mask == 0
                ] = torch.finfo(torch.float32).min

        return combined_attention_mask

    def forward(
            self,
            hidden_states,
            input_ids,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            std=None
    ):
        batch_size, seq_length, _ = hidden_states.shape
        seq_length_with_past = seq_length
        past_key_values_length = 0

        with torch.no_grad():
            inputs_embeds = self.embed_tokens(input_ids)

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
            
        if position_ids is None:
            device = hidden_states.device if hidden_states is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=hidden_states.device
            )
            
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), hidden_states, past_key_values_length
        )


        
        inputs_embeds = inputs_embeds.to(hidden_states.dtype)
        # print(f"inputs_embeds shape: {inputs_embeds.shape}, hidden_states shape: {hidden_states.shape}")
        hidden_states = self.fc(torch.cat((inputs_embeds, hidden_states), dim=-1))

        # print(f"[ModelDynamic Forward] After FC hidden_states.dtype: {hidden_states.dtype}")
        
        all_hidden_states = () if output_hidden_states else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)
                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

        if use_cache:
            return hidden_states, next_decoder_cache

        return hidden_states

 
    def reset_kv(self, client_tag=None):
        """Reset KV cache - supports specific client or global reset"""
        if client_tag is None:
            # Global reset
            self.stable_kv = None
        else:
            # Reset KV cache for specific client
            with self._lock:
                if client_tag in self.client_states:
                    self.client_states[client_tag]['stable_kv'] = None
                    
    def get_client_state(self, client_tag):
        """Get client state; create it if it does not exist."""
        with self._lock:
            if client_tag not in self.client_states:
                self.client_states[client_tag] = {
                    'stable_kv': None,
                    'tree_mask': None,
                    'input_ids_history': None,  # Save complete input_ids history
                    'created_at': time.time()
                }
            return self.client_states[client_tag]
    
    def get_kv_cache(self, client_tag):
        """Get the KV cache for the specified client."""
        client_state = self.get_client_state(client_tag)
        return client_state['stable_kv']
        
    def set_kv_cache(self, client_tag, kv_cache):
        """Set the KV cache for the specified client."""
        client_state = self.get_client_state(client_tag)
        client_state['stable_kv'] = kv_cache
        
    def get_input_ids_history(self, client_tag):
        """Get the input_ids history for the specified client."""
        client_state = self.get_client_state(client_tag)
        return client_state['input_ids_history']

    def get_client_states_memory_usage(self):
        """Calculate the actual memory usage of client_states"""
        total_memory = 0
        
        with self._lock:
            for client_tag, state in self.client_states.items():
                client_memory = 0
                
                # KV Cache memory
                if state['stable_kv'] is not None:
                    for layer_kv in state['stable_kv']:
                        k, v = layer_kv
                        # Bytes per tensor
                        client_memory += k.element_size() * k.nelement()
                        client_memory += v.element_size() * v.nelement()
                
                # Other fields (usually small)
                client_memory += 208  # Python overhead
                
                total_memory += client_memory
                
                print(f"Client {client_tag}: {client_memory / 1024**2:.2f} MB "
                    f"(seq_len: {state['stable_kv'][0][0].shape[2] if state['stable_kv'] else 0})")
        
        print(f"Total client_states memory: {total_memory / 1024**3:.2f} GB")
        return total_memory / 1024**3

    
    def cleanup_client(self, client_tag):
        """Clean up all states for the specified client"""
        with self._lock:
            if client_tag in self.client_states:
                # Clean up GPU memory
                client_state = self.client_states[client_tag]
                if client_state['stable_kv'] is not None:
                    del client_state['stable_kv']
                if client_state['tree_mask'] is not None:
                    del client_state['tree_mask']
                # Remove client state
                del self.client_states[client_tag]
                print(f"✅ Cleaned up all states for client {client_tag}")
    
    def get_detail_time(self):
        """get detail time"""
        with self._lock:
            return self.prepare_time, self.inference_time, self.construction_time, self.inference_time_per_depth

 

    @torch.no_grad()
    def topK_genrate_warmup(self, hidden_states, input_ids, head, logits_processor, backward, accept_length=0, top_k=None, depth=None, client_tag=None, threshold=100):
        """GPU cache warmup function — mirrors topK_genrate but does not update any state.

        Features:
        - Executes all computations of the prepare, inference, and construction stages.
        - Does not update the KV cache (uses the temporary variable `past_key_values_temp`).
        - Does not save `client_state['stable_kv']`.
        - All intermediate results are discarded; used only to warm up the GPU.
        """
        device = hidden_states.device
        
        client_state = self.client_states[client_tag]
        stable_kv = client_state['stable_kv'] 
        
        input_ids = input_ids.to(device)
        max_tokens_possible = top_k * (depth + 1)
        total_tokens = min(self.total_tokens, max_tokens_possible)
        
        # Initialize tree structures
        self.init_tree(top_k)
        
        sample_token = input_ids[:, -1]
        scores_list = []
        parents_list = []
        ss_token = []
        
        input_ids = input_ids[:, 1:]
        len_posi = input_ids.shape[1]
        self.reset()
        
        if stable_kv is not None:
            kv_len = stable_kv[0][0].shape[2]
            out_hidden, past_key_values_temp = self(hidden_states, input_ids=input_ids[:, kv_len:],
                                                     past_key_values=stable_kv, use_cache=True)
        else:
            out_hidden, past_key_values_temp = self(hidden_states, input_ids=input_ids, use_cache=True)
        
        
        last_hidden = out_hidden[:, -1]
        last_headout = head(last_hidden.to(torch.float16))
        last_p = self.logsoftmax(last_headout)
        top = torch.topk(last_p, top_k, dim=-1)
        topk_index, topk_p = top.indices, top.values
        scores = topk_p[0]
        scores_list.append(scores[None])
        parents_list.append(torch.zeros(1, dtype=torch.long, device=scores.device))
        ss_token.append(topk_index)
        input_ids = topk_index
        input_hidden = last_hidden[None].repeat(1, top_k, 1)
        tree_mask = self.tree_mask_init
        topk_cs_index = torch.arange(top_k, device=self.embed_tokens.weight.device)
        
        iteration_count = 1
        
        for i in range(depth):
            iteration_count += 1
            self.tree_mask = tree_mask
            position_ids = len_posi + self.position_ids
            
            out_hidden, past_key_values_temp = self(input_hidden, input_ids=input_ids, 
                                                     past_key_values=past_key_values_temp,
                                                     position_ids=position_ids, use_cache=True)
            len_posi += 1
            
            bias1 = top_k if i > 0 else 0
            bias2 = max(0, i - 1)
            bias = 1 + top_k ** 2 * bias2 + bias1
            parents = (topk_cs_index + bias)
            parents_list.append(parents)
            
            last_headout = head(out_hidden[0].to(torch.float16))
            last_p = self.logsoftmax(last_headout)
            
            top = torch.topk(last_p, top_k, dim=-1)
            topk_index, topk_p = top.indices, top.values
            
            cu_scores = topk_p + scores[:, None]
            topk_cs = torch.topk(cu_scores.view(-1), top_k, dim=-1)
            topk_cs_index, topk_cs_p = topk_cs.indices, topk_cs.values
            scores = topk_cs_p
            
            out_ids = topk_cs_index // top_k
            input_hidden = out_hidden[:, out_ids]
            input_ids = topk_index.view(-1)[topk_cs_index][None]
            
            ss_token.append(topk_index)
            scores_list.append(cu_scores)
            out_ids = out_ids.to(tree_mask.device)
            tree_mask = torch.cat((tree_mask[:, :, out_ids], self.tree_mask_init), dim=3)
        
        scores_list = torch.cat(scores_list, dim=0).view(-1)
        ss_token_list = torch.cat(ss_token, dim=0).view(-1)
        top_scores = torch.topk(scores_list, total_tokens, dim=-1)
        top_scores_index = top_scores.indices
        top_scores_index = torch.sort(top_scores_index).values
        
        draft_tokens = ss_token_list[top_scores_index]
        draft_tokens = torch.cat((sample_token, draft_tokens), dim=0)
        
        draft_parents = torch.cat(parents_list, dim=0)[top_scores_index // top_k].long()
        mask_index = torch.searchsorted(top_scores_index, draft_parents - 1, right=False)
        mask_index[draft_parents == 0] = -1
        mask_index = mask_index + 1
        
        tree_mask_final = build_tree_gpu(mask_index, total_tokens, depth)
        tree_position_ids = torch.sum(tree_mask_final, dim=1) - 1
        
        retrieve_indices = generate_tree_gpu(mask_index, tree_position_ids, total_tokens, depth, logits_processor)
        
        self.tree_mask = None
        torch.cuda.synchronize(device)
        
        return None

    @torch.no_grad()
    def topK_genrate(self, hidden_states, input_ids, head, logits_processor, backward, accept_length=0, top_k=None, depth=None, client_tag=None, threshold=0.77):

        
        prepare_start = torch.cuda.Event(enable_timing=True)
        prepare_end = torch.cuda.Event(enable_timing=True)
        inference_start = torch.cuda.Event(enable_timing=True)
        inference_end = torch.cuda.Event(enable_timing=True)
        construction_start = torch.cuda.Event(enable_timing=True)
        construction_end = torch.cuda.Event(enable_timing=True)
        
        prepare_start.record()
        
        client_state = self.client_states[client_tag]
        stable_kv = client_state['stable_kv']
            
        input_ids = input_ids.to(hidden_states.device)
        max_tokens_possible = top_k * (depth + 1)
        total_tokens = min(self.total_tokens, max_tokens_possible)

        # Initialize tree structures with dynamic top_k
        self.init_tree(top_k)

        sample_token = input_ids[:, -1]  
        scores_list = []
        parents_list = []
        ss_token = []

        input_ids = input_ids[:, 1:]

        len_posi = input_ids.shape[1]
        self.reset()


        if stable_kv is not None:
            kv_len = stable_kv[0][0].shape[2]
            out_hidden, past_key_values = self(hidden_states, input_ids=input_ids[:, kv_len:],
                                                past_key_values=stable_kv, use_cache=True)

        else:
            out_hidden, past_key_values = self(hidden_states, input_ids=input_ids, use_cache=True)

        client_state['stable_kv'] = past_key_values

        last_hidden = out_hidden[:, -1]

        last_headout = head(last_hidden.to(torch.float16))  
        last_p = self.logsoftmax(last_headout)
        top = torch.topk(last_p, top_k, dim=-1)
        topk_index, topk_p = top.indices, top.values
        scores = topk_p[0]
        scores_list.append(scores[None])
        parents_list.append(torch.zeros(1, dtype=torch.long, device=scores.device))
        ss_token.append(topk_index)
        input_ids = topk_index
        input_hidden = last_hidden[None].repeat(1, top_k, 1)
        tree_mask = self.tree_mask_init
        topk_cs_index = torch.arange(top_k, device=self.embed_tokens.weight.device)
        
        prepare_end.record()
        inference_start.record()
        iteration_count = 1
        
        depth_events = []
        for d in range(depth):
            depth_events.append({
                'start': torch.cuda.Event(enable_timing=True),
                'end': torch.cuda.Event(enable_timing=True)
            })
    

        for i in range(depth):
            depth_events[i]['start'].record()
            iteration_count += 1
            self.tree_mask = tree_mask
            position_ids = len_posi + self.position_ids

            out_hidden, past_key_values = self(input_hidden, input_ids=input_ids, past_key_values=past_key_values,
                                               position_ids=position_ids, use_cache=True)
            len_posi += 1
            
            bias1 = top_k if i > 0 else 0
            bias2 = max(0, i - 1)
            bias = 1 + top_k ** 2 * bias2 + bias1
            parents = (topk_cs_index + bias)
            parents_list.append(parents)

            last_headout = head(out_hidden[0].to(torch.float16))
            last_p = self.logsoftmax(last_headout)

            top = torch.topk(last_p, top_k, dim=-1)
            topk_index, topk_p = top.indices, top.values

            cu_scores = topk_p + scores[:, None]

            # # 1. calculate entropy for each path
            # entropy = -torch.sum(torch.exp(last_p) * last_p, dim=-1)  # [top_k]
            
            # # 2. Calculate the path weight based on cumulative probability
            # # scores: [top_k] - current cumulative log probabilities
            # path_probs = torch.exp(scores)  # convert log probs to probs
            # path_probs = path_probs / path_probs.sum()  # normalize

            # # 3. calculate weighted entropy
            # weighted_entropy = (entropy * path_probs).sum().item()
            # print("Entropy:", weighted_entropy)

            topk_cs = torch.topk(cu_scores.view(-1), top_k, dim=-1)
            topk_cs_index, topk_cs_p = topk_cs.indices, topk_cs.values
            scores = topk_cs_p

            out_ids = topk_cs_index // top_k 
            
            # 1. calculate entropy for each path
            # entropy = -torch.sum(torch.exp(last_p) * last_p, dim=-1)  # [top_k]
            
            # 2. find highest accumulated probability path (use out_ids which is already computed)
            # best_path_idx = out_ids[0].item()  # best path index in original top_k paths
            # best_path_entropy = entropy[best_path_idx].item()

            input_hidden = out_hidden[:, out_ids]

            input_ids = topk_index.view(-1)[topk_cs_index][None]

            ss_token.append(topk_index)
            scores_list.append(cu_scores)
            out_ids = out_ids.to(tree_mask.device)
            tree_mask = torch.cat((tree_mask[:, :, out_ids], self.tree_mask_init), dim=3)

            # 4. early stop condition
            # if math.sqrt(best_path_entropy) > threshold: break
            
            depth_events[i]['end'].record()
        
        inference_end.record()
        construction_start.record()
        
        scores_list = torch.cat(scores_list, dim=0).view(-1)
        score_time = time.time()
        ss_token_list = torch.cat(ss_token, dim=0).view(-1)
        top_scores = torch.topk(scores_list, total_tokens, dim=-1)
        top_scores_index = top_scores.indices
        top_scores_index = torch.sort(top_scores_index).values

        draft_tokens = ss_token_list[top_scores_index]
        draft_tokens = torch.cat((sample_token, draft_tokens), dim=0)

        draft_parents = torch.cat(parents_list, dim=0)[top_scores_index // top_k].long()  
        mask_index = torch.searchsorted(top_scores_index, draft_parents - 1, right=False)
        mask_index[draft_parents == 0] = -1
        mask_index = mask_index + 1
        mask_index_list = mask_index.tolist() 

        tree_mask = torch.eye(total_tokens + 1).bool()
        tree_mask[:, 0] = True
        for i in range(total_tokens):
            tree_mask[i + 1].add_(tree_mask[mask_index_list[i]])

        tree_position_ids = torch.sum(tree_mask, dim=1) - 1
        tree_mask = tree_mask.float()[None, None]
        draft_tokens = draft_tokens[None]

        del parents_list, scores_list, ss_token, ss_token_list, draft_parents

        max_depth = torch.max(tree_position_ids) + 1

        noleaf_index = torch.unique(mask_index).tolist()  
        noleaf_num = len(noleaf_index) - 1
        leaf_num = total_tokens - noleaf_num
        retrieve_indices = torch.zeros(leaf_num, max_depth.item(), dtype=torch.long) - 1
        retrieve_indices = retrieve_indices.tolist()

        rid = 0
        position_ids_list = tree_position_ids.tolist()
        for i in range(total_tokens + 1):
            if i not in noleaf_index:
                cid = i
                depth = position_ids_list[i]
                for j in reversed(range(depth + 1)):
                    retrieve_indices[rid][j] = cid
                    cid = mask_index_list[cid - 1] 
                rid += 1

        if logits_processor is not None:
            maxitem = total_tokens + 5

            def custom_sort(lst):
                sort_keys = []
                for i in range(len(lst)):
                    sort_keys.append(lst[i] if lst[i] >= 0 else maxitem)
                return sort_keys

            retrieve_indices = sorted(retrieve_indices, key=custom_sort)

        retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)
        del mask_index, noleaf_index, noleaf_num, leaf_num, max_depth, rid
        tree_position_ids = tree_position_ids.to(hidden_states.device)
        
        construction_end.record()
        
        torch.cuda.synchronize()
        self.prepare_time = prepare_start.elapsed_time(prepare_end) / 1000.0 
        self.inference_time = inference_start.elapsed_time(inference_end) / 1000.0
        self.construction_time = construction_start.elapsed_time(construction_end) / 1000.0
        
        self.inference_time_per_depth = [
            depth_events[i]['start'].elapsed_time(depth_events[i]['end']) / 1000.0
            for i in range(len(depth_events))
        ]
        
        self.last_inference_time = time.time()
        
        return draft_tokens, retrieve_indices, tree_mask, tree_position_ids, draft_tokens.shape[1], iteration_count


def build_tree_gpu(mask_index, total_tokens, max_depth):
    device = mask_index.device
    tree_mask = torch.eye(total_tokens + 1, dtype=torch.bool, device=device)
    tree_mask[:, 0] = True  
    start_time = time.time()
    if max_depth == 0:  
        return tree_mask

    for _ in range(max_depth + 1):
        children = torch.arange(1, total_tokens + 1, device=device)
        parents = mask_index[:total_tokens].long()
        updates = tree_mask[parents] 
        tree_mask[children] |= updates 
        
    return tree_mask



@torch.no_grad()
def generate_tree_gpu(
    mask_index: torch.Tensor,
    tree_position_ids: torch.Tensor,
    total_tokens: int,
    max_depth: int,
    logits_processor=None
):
    device = mask_index.device
    
    noleaf_mask = torch.zeros(total_tokens + 1, dtype=torch.bool, device=device)
    noleaf_mask.scatter_(0, mask_index[mask_index >= 0], True) 
    leaf_mask = ~noleaf_mask
    
    leaf_indices = torch.where(leaf_mask)[0]
    retrieve_indices = -torch.ones((len(leaf_indices), max_depth), 
                                 dtype=torch.long, device=device)
    
    for depth in range(max_depth):
        valid = tree_position_ids[leaf_indices] >= depth
        retrieve_indices[valid, depth] = leaf_indices[valid]
        leaf_indices = torch.where(leaf_indices > 0, 
                                 mask_index[leaf_indices - 1], 
                                 -1)

    if logits_processor is not None:
        sort_key = torch.where(retrieve_indices >= 0, 
                             retrieve_indices, 
                             total_tokens + 5)
        sorted_idx = torch.argsort(sort_key[:, 0]) 
        retrieve_indices = retrieve_indices[sorted_idx]
    
    return retrieve_indices
