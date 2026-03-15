import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel( # Runs in parallel — one GPU thread per token.
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr, # note: in Triton kernels every tensor argument automatically becomes a pointer to its first element 
    D: tl.constexpr, # this tells the Triton compiler that the value of D will not change during the execution of the kernel. 
):
    idx = tl.program_id(0) # which kernel instance is this thread running? corresponds to the token index in the batch dimension.
    
    # Load this token's destination slot in the KV cache.
    slot = tl.load(slot_mapping_ptr + idx) # python slot_mapping[idx]; slot_mapping_ptr is a raw pointer to the start of the slot_mapping array in GPU memory. To get the slot for token idx, you need to read the element at position idx in that array
    if slot == -1: return
    
    # Read this token's full K & V vector
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D) # slot * D jumps to the right position in the flat cache memory for this token's K/V vector. Each slot = one token's K/V storage location, corresponds to a contiguous segment of size D in the flat cache memory, where this token's K/V will be stored.
    
    # Write K and V into their designated slot in the KV cache. slot * D jumps to the right position in the flat cache memory.
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape # N = total # of tokens in this batch; num_heads = number of kv heads; head_dim = dimension per head
    D = num_heads * head_dim # embedding dimension
    assert key.stride(-1) == 1 and value.stride(-1) == 1 # shape: (N, num_kv_heads, head_dim) 
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D # shape: (num_kvcache_blocks, block_size, num_kv_heads, head_dim), 代表了大KV cache中一个layer的所有blocks
    assert slot_mapping.numel() == N # numel() = 'number of elements' in the tensor 
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache # FlashAttention needs to see the entire key/value history — both cached prefix and new tokens
            o = flash_attn_varlen_func(q, k, v, # FlashAttention function, handles variable-length packed sequences
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache, # optimized for decode phase, 1 query token per sequence attending to a long KV cache
                                        cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                        softmax_scale=self.scale, causal=True)
        return o
