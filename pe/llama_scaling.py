"""
Source: https://github.com/ymcui/Chinese-LLaMA-Alpaca-2
Author: ymcui
License: Apache-2.0 license
"""
import torch
from torch import nn
from typing import Optional, Tuple, Union
import transformers
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, rotate_half
import math


STORE_KV_BEFORE_ROPE = False
USE_MEM_EFF_ATTENTION = False
ALPHA = 1.0
AUTO_COEFF = 1.0
SCALING_FACTOR = None


def apply_rotary_pos_emb_single(q, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    return q_embed


old_init = transformers.models.llama.modeling_llama.LlamaRotaryEmbedding.__init__


def _set_cos_sin_cache(self, seq_len, device, dtype):
    self.max_seq_len_cached = seq_len
    t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.float32)
    t = t / self.scaling_factor

    freqs = torch.einsum("i,j->ij", t, self.ntk_inv_freq.to(device))
    # Different from paper, but it uses a different permutation in order to obtain the same calculation
    emb = torch.cat((freqs, freqs), dim=-1)
    self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
    self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)


def adaptive_ntk_init(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=None):
    self.alpha = ALPHA
    if SCALING_FACTOR is None:
        self.scaling_factor = scaling_factor or 1.0
    else:
        self.scaling_factor = SCALING_FACTOR
    if isinstance(ALPHA,(float,int)):
        base = base * ALPHA ** (dim / (dim-2))
        self.base = base
    elif ALPHA=='auto':
        self.base = base
    else:
        raise ValueError(ALPHA)
    old_init(self, dim, max_position_embeddings, base, device)
    self.ntk_inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))

    self._set_cos_sin_cache = _set_cos_sin_cache
    self._set_cos_sin_cache(
        self, seq_len=max_position_embeddings, device=self.ntk_inv_freq.device, dtype=torch.get_default_dtype()
    )


def adaptive_ntk_forward(self, x, seq_len=None):
    if seq_len > self.max_seq_len_cached:
        if isinstance(self.alpha,(float,int)):
            self._set_cos_sin_cache(self, seq_len=seq_len, device=x.device, dtype=x.dtype)
        elif self.alpha=='auto':
            t = torch.arange(seq_len, device=x.device, dtype=torch.float32)
            t = t / self.scaling_factor
            dim = self.dim
            alpha = (seq_len / (self.max_position_embeddings/2) - 1) * AUTO_COEFF
            base = self.base * alpha ** (dim / (dim-2))
            ntk_inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(x.device) / dim ))

            freqs = torch.einsum("i,j->ij", t, ntk_inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            cos_cached = emb.cos()[None, None, :, :]
            sin_cached = emb.sin()[None, None, :, :]
            return (
                cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
                sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype)
            )
    return (
        self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype)
    )


def apply_ntk_scaling_patch(alpha: Union[float,str], scaling_factor: Optional[float] = None):
    global ALPHA
    global SCALING_FACTOR
    ALPHA = alpha
    SCALING_FACTOR = scaling_factor
    try:
        ALPHA = float(ALPHA)
    except ValueError:
        if ALPHA!="auto":
            raise ValueError(f"Alpha can only be a float or 'auto', but given {ALPHA}")
    print(f"Apply NTK scaling with ALPHA={ALPHA}")
    if scaling_factor is None:
        print(f"The value of scaling factor will be read from model config file, or set to 1.")
    else:
        print(f"Warning: scaling factor is set to {SCALING_FACTOR}. \
              If you set the value by hand, do not forget to update \
              max_position_embeddings in the model config file.")

    transformers.models.llama.modeling_llama.LlamaRotaryEmbedding.__init__ = adaptive_ntk_init
    if hasattr(transformers.models.llama.modeling_llama,'LlamaLinearScalingRotaryEmbedding'):
        transformers.models.llama.modeling_llama.LlamaLinearScalingRotaryEmbedding.__init__ = adaptive_ntk_init
    transformers.models.llama.modeling_llama.LlamaRotaryEmbedding.forward = adaptive_ntk_forward