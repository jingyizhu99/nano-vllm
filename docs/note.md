# nano-vllm Study Notes

## Layers (layers/)

### Why custom layers?
Standard HuggingFace layers don't know about paged KV cache, tensor parallelism, or packed weights.
nano-vllm replaces them with versions that have the same math but are:
- **KV-cache-aware**: write K/V into paged slots, read via block_tables
- **Tensor-parallel**: weights sharded across GPUs, NCCL all-reduce after matmul
- **Optimized**: fused CUDA/Triton kernels instead of standard PyTorch ops

---

### attention.py

#### `store_kvcache_kernel` (Triton GPU kernel)
- Runs `N` parallel threads — one per token (`idx = tl.program_id(0)`)
- Each thread: reads `slot = slot_mapping[idx]`, skips if `slot == -1` (padding)
- Reads token's full K vector: `key_offsets = idx * key_stride + tl.arange(0, D)`
  - `idx * key_stride` → jump to token idx's start in flat memory
  - `tl.arange(0, D)` → read all D = num_kv_heads * head_dim elements
- Writes to KV cache: `cache_offsets = slot * D + tl.arange(0, D)`
  - `slot * D` → jump to destination slot in flat KV cache memory
  - Different from `key_offsets` — source is indexed by batch position, destination by physical slot
- `tl.load` / `tl.store` are vectorized — read/write all D values in one instruction

#### Tensor shapes & strides
- `key.shape = (N, num_kv_heads, head_dim)` — computed K vectors for current batch (not a weight matrix)
- `k_cache.shape = (num_blocks, block_size, num_kv_heads, head_dim)`
- `key.stride(1) = head_dim` — step over one head
- `k_cache.stride(1) = D = num_kv_heads * head_dim` — step over one token slot (bigger because 2 more dims after)
- `stride(-1) == 1` asserts last dim is contiguous so `tl.arange` reads consecutive memory

#### `Attention.forward`
```
1. store_kvcache(k, v, slot_mapping)     ← always: write new K/V to paged cache

2a. PREFILL:
    if prefix cache: k, v = k_cache, v_cache   ← swap to full cache so queries attend full history
    flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, block_table)
    - handles variable-length packed sequences (no padding)
    - seqlen_q < seqlen_k when prefix cache hit

2b. DECODE:
    flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache, cache_seqlens, block_table)
    - optimized for 1 query token per sequence attending to long KV cache
    - q.unsqueeze(1): (bs, heads, dim) → (bs, 1, heads, dim)
```

#### Why two FlashAttention functions?
- `flash_attn_varlen_func` — prefill: many tokens/seq, variable lengths, packed
- `flash_attn_with_kvcache` — decode: 1 token/seq, reads from paged KV cache
- Same math (Q·K^T softmax V) but different memory access patterns → different optimizations

#### What is a slot?
- `slot = block_id * block_size + position_within_block` — flat index into KV cache
- One slot = one token's K/V storage location
- `slot_mapping` maps batch token index → physical KV cache slot (scatter write)

---

### embed_head.py

#### `VocabParallelEmbedding` — input embedding (token ID → vector)
- Vocab table split across GPUs by token ID range
- `num_embeddings` = vocab size (e.g. 128000) — total number of unique tokens
- GPU 0 owns IDs `[0, vocab/tp)`, GPU 1 owns `[vocab/tp, 2*vocab/tp)`, etc.
- `weight.shape = (num_embeddings_per_partition, embedding_dim)` per GPU
- `weight_loader`: same as ColumnParallel — slices `tp_rank`'s shard from checkpoint

**Forward** (multi-GPU):
```python
mask = (x >= vocab_start_idx) & (x < vocab_end_idx)  # which tokens this GPU owns
x = mask * (x - vocab_start_idx)   # global ID → local index; out-of-range → 0 (safe dummy lookup)
y = F.embedding(x, self.weight)    # lookup — wrong tokens produce garbage values
y = mask.unsqueeze(1) * y          # zero out embeddings for tokens this GPU doesn't own
dist.all_reduce(y)                 # sum across GPUs → correct embedding (only 1 GPU was non-zero per token)
```

#### `ParallelLMHead` — output projection (hidden state → logits)
- Inherits `VocabParallelEmbedding` — reuses vocab table weight (tied embeddings)
- Reverse direction: hidden states → logit scores over vocab
- Prefill: only last token of each sequence needs logits
  - `last_indices = cu_seqlens_q[1:] - 1` → last token index per sequence in flat packed tensor
- `logits = F.linear(x, self.weight)` → `(num_seqs, vocab_size_per_gpu)`
- Uses `dist.gather` (not all-reduce) — only rank 0 needs full logits for sampling
  - `gather` → all partial logits collected to rank 0 → concat → `(num_seqs, full_vocab_size)`
  - Workers get `None`

#### all-reduce vs gather
```
all-reduce → all GPUs get the result  → used mid-network (next layer needs it on all GPUs)
gather     → only rank 0 gets result  → used at end (only rank 0 samples tokens)
```

#### `self.weight.weight_loader = self.weight_loader`
- Attaches the class's `weight_loader` method as an attribute on the parameter object
- `load_model()` does `getattr(param, "weight_loader", default_weight_loader)` — this is how it finds the custom loader
- Makes loading behavior self-contained on the parameter — `load_model()` needs no knowledge of TP internals

---

### rotary_embedding.py

#### What is RoPE?
- Encodes token position by **rotating** Q and K vectors by a position-dependent angle
- Tokens closer in position have more similar rotations → higher attention scores
- Applied to Q and K only — V is never rotated (position only matters for attention weights Q·K, not values)

#### `__init__` — precompute cos/sin table
```python
inv_freq = 1.0 / (base ** (arange(0, rotary_dim, 2) / rotary_dim))
# rotary_dim/2 geometrically spaced frequencies
# low index = high freq (sensitive to nearby positions)
# high index = low freq (sensitive to long-range positions)

freqs = einsum("i,j->ij", positions, inv_freq)  # outer product: (max_pos, rotary_dim/2)
cache = cat(cos, sin).unsqueeze(1)               # (max_pos, 1, rotary_dim) — 1 for head broadcasting
register_buffer("cos_sin_cache", cache, persistent=False)  # non-trainable, moves to GPU, not saved
```

#### `apply_rotary_emb` — 2D rotation on pairs
```python
x1, x2 = chunk(x, 2, dim=-1)   # split head_dim in half → rotary_dim/2 pairs
y1 = x1*cos - x2*sin            # 2D rotation:  [cos -sin] [x1]
y2 = x2*cos + x1*sin            #               [sin  cos] [x2]
```
Each pair `(x1[i], x2[i])` uses a different frequency — head_dim must be split because rotation operates on pairs of values.

#### `forward` — lookup and apply
```python
cos_sin = cos_sin_cache[positions]   # lookup by explicit positions tensor
query = apply_rotary_emb(query, cos, sin)
key   = apply_rotary_emb(key,   cos, sin)
```
Uses explicit `positions` tensor (not sequential 0,1,2) — critical for decode where position = `len(seq)-1`.

#### `rotary_dim == head_size`
Modern models rotate all head dimensions (not partial). The assert enforces this — simplifies code (no need to handle partial rotation).

#### `@lru_cache(1)` on `get_rope`
All attention layers share the same RoPE config → cache size 1 returns the same instance on repeated calls, avoids recomputing cos/sin table per layer.

---

### layernorm.py

#### What is RMSNorm?
Normalizes hidden states to prevent activations growing too large across layers:
```
x = x / sqrt(mean(x²) + eps) * weight
```
Simpler than LayerNorm (no mean subtraction). Used by Llama, Qwen, etc.

- `weight` (γ): learned per-dimension scale, initialized to 1 — lets model undo normalization if needed
- `eps`: prevents division by zero
- Upcasts to float32 for numerical stability, casts back to bfloat16 after

#### Residual stream
Each transformer block: `output = sublayer(norm(x)) + x`
The `+ x` is the residual connection — carries input through, helps gradient flow.

nano-vllm carries the residual **unnormalized** and fuses the addition into the norm:

```python
# each decoder layer:
hidden, residual = norm(hidden, residual)   # fused: hidden = Norm(hidden + residual)
                                            #        residual = hidden + residual (unnormalized)
hidden = attention(hidden)
hidden, residual = norm(hidden, residual)   # fused again
hidden = mlp(hidden)
# residual accumulates: input + attn_out + mlp_out + ...
```

#### `add_rms_forward(x, residual)`
```python
x = x + residual          # x = sublayer_output + running_sum
residual = x              # save unnormalized sum for next layer
x = Norm(x)               # normalize
return x, residual
```

#### `@torch.compile` on rms_forward / add_rms_forward
JIT-compiles to fused CUDA kernel — pow, mean, rsqrt, mul fused into one GPU kernel, intermediate results stay in registers (no GPU memory writes between ops).

---

### activation.py

#### `SiluAndMul` — SwiGLU activation for MLP
```python
x, y = x.chunk(2, -1)    # split fused gate+up output in half
return F.silu(x) * y     # SwiGLU: silu(gate) * up
```
Input comes from `MergedColumnParallelLinear` which fuses gate and up projections into one matmul. This splits them and applies the gating. `@torch.compile` fuses the chunk + silu + mul into one kernel.

---

### Triton vs CUDA vs @torch.compile
```
CUDA C++         → lowest level, manual thread/memory management, hardest, fastest ceiling
Triton DSL       → Python-like GPU DSL, compiles to PTX, handles memory coalescing automatically
@torch.compile   → highest level, compiler auto-fuses PyTorch ops, easiest
```
- `store_kvcache_kernel` → Triton (custom scatter-write, no PyTorch built-in)
- `rms_forward`, `apply_rotary_emb`, `SiluAndMul` → `@torch.compile` (standard ops, auto-fused)
- FlashAttention → CUDA C++ (written by flash-attn library, maximum performance)

---

### linear.py

#### Class hierarchy
```
LinearBase
  ├── ReplicatedLinear          — no TP, full weight on every GPU
  ├── ColumnParallelLinear      — split output dim (tp_dim=0)
  │     ├── MergedColumnParallelLinear  — multiple fused projections (e.g. gate+up)
  │     └── QKVParallelLinear           — fused Q,K,V with GQA support
  └── RowParallelLinear         — split input dim (tp_dim=1), needs all-reduce
```

#### `LinearBase`
- Allocates `weight` as `nn.Parameter(torch.empty(output_size, input_size))`
- Attaches `weight_loader` directly onto the parameter: `self.weight.weight_loader = self.weight_loader`
- This is how `load_model()` knows which loader to call per parameter

#### `param.data.copy_(loaded_weight)`
- `param` = GPU tensor allocated but uninitialized (`torch.empty`)
- `loaded_weight` = tensor read from `.safetensors` on CPU
- `.data` = raw tensor without gradient tracking
- `.copy_()` = in-place copy, handles CPU→GPU transfer automatically
- Can't use `param.data = loaded_weight` — that replaces the pointer instead of filling GPU memory

#### `tp_dim` — which dimension to split
- `tp_dim=0` → split rows (output dimension) → ColumnParallel
- `tp_dim=1` → split columns (input dimension) → RowParallel
- Same `weight_loader` code works for both by passing different `tp_dim`

#### `ColumnParallelLinear`
- Each GPU allocates `output_size // tp_size` rows
- `weight_loader`: `narrow(tp_dim, tp_rank * shard_size, shard_size)` slices correct rows from checkpoint
- No all-reduce needed — each GPU's partial output is used independently downstream
- Used for: Q, K, V projections, MLP gate/up

#### `RowParallelLinear`
- Each GPU allocates `input_size // tp_size` columns
- Each GPU computes a partial sum → must all-reduce to get full result
- Bias only added by rank 0 (to avoid adding it `tp_size` times after all-reduce)
- Used for: attention output projection, MLP down

#### `MergedColumnParallelLinear` — fused sub-projections (e.g. gate + up)
- `output_sizes = [intermediate_size, intermediate_size]` — two sub-projections packed into one weight
- `weight_loader(param, loaded_weight, loaded_shard_id: int)`:
  - `shard_offset = sum(output_sizes[:id]) // tp_size` — where this sub-proj starts in local param
  - `shard_size = output_sizes[id] // tp_size` — how many rows it occupies
  - `param_data.narrow(...)` → destination slice in fused weight
  - `loaded_weight.chunk(tp_size, tp_dim)[tp_rank]` → this GPU's chunk of full checkpoint weight
- Called twice (once per sub-projection), fills different regions of the same fused param

#### `QKVParallelLinear` — fused Q, K, V for attention
- Q, K, V stored separately in checkpoint but fused into one weight for efficiency
- Handles GQA: Q has more heads than K, V (`total_num_heads` vs `total_num_kv_heads`)
- `output_size = (total_num_heads + 2 * total_num_kv_heads) * head_size`
- Local param layout per GPU:
  ```
  [Q shard | K shard | V shard]
   num_heads*head_size | num_kv_heads*head_size | num_kv_heads*head_size
  ```
- `weight_loader(param, loaded_weight, loaded_shard_id: str)` — called 3x with "q", "k", "v"
  - Each call computes its `shard_offset` and `shard_size` based on head counts
  - Q: offset=0, size=num_heads*head_size
  - K: offset=num_heads*head_size, size=num_kv_heads*head_size
  - V: offset=num_heads*head_size + num_kv_heads*head_size, size=num_kv_heads*head_size

#### SwiGLU (gate + up projections)
- Modern LLMs use gated MLP instead of plain MLP
- `output = F.silu(gate_proj(x)) * up_proj(x)` → down_proj
- Gate acts as learned filter controlling information flow
- gate and up have same shape → fused into one matmul via `MergedColumnParallelLinear`

#### `.narrow(dim, start, length)` vs slicing
- Returns a **view** (no memory copy) — efficient for large weight matrices
- Works cleanly for any dimension: `narrow(0,...)` = rows, `narrow(1,...)` = columns

---

## General Concepts

### Tokenizer
- `AutoTokenizer.from_pretrained(config.model, use_fast=True)` loads a pre-trained tokenizer
- Every LLM has its own tokenizer (incompatible across models)
- Tokenizer converts text → token IDs (encode) and token IDs → text (decode)
- `use_fast=True` uses Rust-based implementation (much faster)
- Underlying algorithms: BPE (most common) or SentencePiece

### Temperature
- Controls randomness of token sampling
- `logits /= temperature` before softmax
- `temp < 1` → more deterministic; `temp > 1` → more random; `temp → 0` → greedy
- Each sequence in a batch can have a different temperature

### Attention Heads vs KV Heads
- `num_attention_heads` = number of Q heads (always)
- `num_key_value_heads` = number of KV heads
- **MHA**: Q heads == KV heads (every Q has its own K,V)
- **GQA**: Q heads > KV heads (groups of Q heads share one K,V) — modern standard (Llama3, Qwen)
- **MQA**: all Q heads share a single K,V — most aggressive KV cache reduction
- `head_dim = hidden_size / num_attention_heads`
- `hidden_size` = width of each token vector (e.g. 4096) — NOT the number of layers
- `num_hidden_layers` = depth, how many transformer blocks stacked (e.g. 32)

### Gumbel-max Sampling
- Mathematically equivalent to `torch.multinomial` but faster
- `probs / Exponential(1) noise → argmax` — tokens with higher prob more likely to win but not guaranteed
- `argmax` is a single parallel GPU op vs multinomial's sequential scan

---

## Request Lifecycle (llm_engine.py)

### Flow
```
generate(prompts, sampling_params)
  → add_request(): tokenize → Sequence → scheduler.waiting
  → step() loop:
      → scheduler.schedule() → (seqs, is_prefill)
      → model_runner.call("run", seqs, is_prefill) → token_ids
      → scheduler.postprocess(): append token, check EOS/max_tokens
  → tokenizer.decode() → return text
```

### num_tokens sign convention
- `num_tokens > 0` → prefill phase; value = total prompt tokens processed
- `num_tokens < 0` → decode phase; magnitude = number of sequences (each generates exactly 1 token)
- Dual purpose: encodes both the count (magnitude) and phase (sign) in one value

### Prefill vs Decode
- **Prefill**: processes all prompt tokens in one forward pass, builds KV cache, generates 1st token
  - All tokens processed in parallel → fast
  - `input_ids` = all uncached prompt tokens concatenated across sequences
- **Decode**: feeds 1 token per sequence per step, reads KV cache for context
  - Sequential (one token at a time) → slow, memory-bandwidth-bound
  - `input_ids` = only `last_token` per sequence
- `compute_logits` during prefill extracts only the last token's hidden state per sequence:
  `last_indices = cu_seqlens_q[1:] - 1` → still outputs 1 token per sequence

---

## Scheduler (scheduler.py)

### Two phases
- **Prefill**: moves sequences from `waiting` → `running`, allocates KV cache blocks
  - Constraints: `max_num_seqs`, `max_num_batched_tokens`
  - `num_batched_tokens += len(seq) - seq.num_cached_tokens` (only uncached tokens count)
- **Decode**: all `running` sequences generate 1 token each
  - Constraint: must have 1 free block per sequence; preempts if memory tight

---

## Block Manager (block_manager.py)

### Block (logical concept, not GPU memory)
- `Block` is pure CPU-side bookkeeping — it does NOT directly represent a GPU memory segment
- `block_id` is just an index into `kv_cache[:, :, block_id, :, :, :]` — the actual GPU tensor
- Fields: `block_id`, `ref_count` (sharing count), `hash` (-1 = partial/unfinalised), `token_ids` (for verification)
- Actual GPU memory is written by attention kernel via `slot_mapping`, read via `block_tables`

### block_table
- Per-sequence list of physical block IDs: `[2, 5, 11]` means logical block 0 → physical block 2, etc.
- Set during `allocate()`, grown during `may_append()`, cleared during `deallocate()`
- Acts as a **page table** for KV cache (same concept as virtual memory)

### num_cached_tokens
- Tracks how many prefix tokens already have K/V in GPU memory (prefix cache hit)
- Set in `allocate()` when block hash matches an existing cached block (`+= block_size`)
- Can only be a prefix — once a cache miss occurs, all subsequent blocks must be fresh
- Reset to 0 on `deallocate()`

### Prefix caching
- Blocks are identified by **chained hash**: block i's hash = hash(its tokens + previous block's hash)
- Chaining ensures two blocks with identical content but different prefixes get different hashes
- If a new sequence shares a prefix with a cached sequence, those blocks are reused (ref_count++)
- Only **full** blocks are hashed and eligible for sharing — partial last block always has `hash == -1`
- `hash_to_block_id` is never cleaned up on deallocation (lazy eviction)
  - A freed block can be reclaimed by the next sequence needing the same prefix
  - Safety: `block.token_ids != token_ids` check catches stale hash entries

### Block lifecycle during decode (may_append)
- Called after every `append_token()` — `len(seq)` grows by 1 each step
- `len(seq) % block_size == 1` → just entered a new block → allocate a fresh block
- `len(seq) % block_size == 0` → just filled a block → finalize with chained hash, register for prefix sharing
- `else` → mid-block, nothing to do
- Hash chaining: `prefix = blocks[block_table[-2]].hash` (or -1 if first block)

### deallocate
- Iterates `block_table` in reverse (reverse construction order — defensive convention)
- Decrements `ref_count`; only returns block to `free_block_ids` when `ref_count == 0`
- Shared prefix blocks stay alive as long as any sequence still references them

---

## Model Runner (model_runner.py)

### __init__ sequence (every GPU rank runs this)
```
init NCCL → set GPU device → load model weights → warmup_model()
→ allocate_kv_cache() → capture_cudagraph() (if not enforce_eager)
→ rank 0: create shared memory → return → serve requests
  rank 1+: attach shared memory → loop() → wait for commands forever
```

### Tensor Parallelism
- Each GPU owns a vertical slice of KV heads: `num_kv_heads // world_size` heads per GPU
- During prefill and decode, each GPU computes K/V only for its assigned heads
- Each GPU has its own `kv_cache` tensor storing only its heads' K/V values
- After attention, all-reduce combines partial results → all GPUs get identical hidden states
- By the time logits are computed, all GPUs have identical values → only rank 0 samples

### IPC (Inter-Process Communication) for multi-GPU
- Rank 0 writes `(method_name, args)` to shared memory, sets `mp.Event` to wake workers
- Workers block on `event.wait()`, read args, execute same method
- All ranks run the model forward pass simultaneously → NCCL all-reduce syncs them

### allocate_kv_cache
- Runs after `warmup_model()` to measure peak GPU memory used by a forward pass
- Available memory = `total × gpu_memory_utilization - used - peak + current`
- `block_bytes = 2 × num_layers × block_size × num_kv_heads × head_dim × dtype_bytes`
- `num_kvcache_blocks = available_memory // block_bytes`
- Allocates one big tensor: `kv_cache[K/V, layer, block_id, token, head, dim]`
- Assigns each attention layer its slice: `module.k_cache = kv_cache[0, layer_id]`

### prepare_prefill(seqs)
Converts a batch of sequences into flat tensors for the model:
- `input_ids`: uncached tokens only (`seq[num_cached_tokens:]`)
- `positions`: absolute positions of uncached tokens (for RoPE)
- `cu_seqlens_q`: cumulative query lengths `[0, q1, q1+q2, ...]` — FlashAttention sequence boundaries
- `cu_seqlens_k`: cumulative key lengths (includes cached prefix) — `seqlen_k = seqlen`
- `seqlen_q < seqlen_k` when prefix cache hit: queries are shorter than keys
- `slot_mapping`: where to write new K/V — `block_table[i] * block_size + offset`
- `block_tables`: 2D `(num_seqs, max_num_blocks)` padded with `-1`, for FlashAttention to read cached K/V

### prepare_decode(seqs)
- `input_ids`: only `seq.last_token` per sequence (1 token per seq)
- `positions`: `len(seq) - 1` (absolute position of new token)
- `slot_mapping`: `block_table[-1] * block_size + last_block_num_tokens - 1`
- `context_lens`: `len(seq)` per sequence (how many K/V entries in cache)
- `block_tables`: always built

### run_model(input_ids, positions, is_prefill)
- **Eager path** (prefill, enforce_eager, bs>512): `model(input_ids, positions)` → `compute_logits()`
- **CUDA graph path** (decode, bs≤512):
  1. Round up to nearest captured graph size
  2. Update `graph_vars` tensor contents (input_ids, positions, slot_mapping, context_lens, block_tables)
  3. `graph.replay()` — replays recorded kernel sequence on updated data
  4. Return `graph_vars["outputs"][:bs]` → `compute_logits()`

### CUDA graph concept
- **Without**: CPU launches each GPU kernel individually (~5-10μs × hundreds of kernels = ~3ms/step)
- **With**: record kernel sequence once, replay instantly (~0.1ms/step)
- Constraint: tensor shapes and GPU memory addresses must be fixed (baked into graph at capture time)
- Only used for decode — prefill has variable sequence lengths

### capture_cudagraph
- Captures one graph per batch size in `graph_bs = [1, 2, 4, 8, 16, 32, ...]`
- Why multiple sizes: each graph hardcodes tensor slice addresses (e.g. `[:4]`) — can't replay for different size
- At runtime, round up: `graph = graphs[smallest graph_bs >= actual bs]` — extra slots compute garbage, discarded
- Why reversed (largest first): first capture initializes `graph_pool` — must be sized for the largest graph
- All graphs share one `graph_pool` (memory pool for intermediate tensors) — saves GPU memory since only one replays at a time
- `graph_vars` = the fixed tensors whose contents are updated before each `replay()`
