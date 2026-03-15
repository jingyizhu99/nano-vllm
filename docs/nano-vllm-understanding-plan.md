# nano-vllm Architecture Study Plan

---

## Phase 1: The Macro Engine
**Goal:** Understand the lifecycle of a request from API entry to token generation.

**Files:**
- `example.py`
- `nanovllm/llm.py`
- `nanovllm/engine/llm_engine.py`

- [x] Trace `LLMEngine.add_request()` — how prompt text becomes a `Sequence`
- [x] Study `LLMEngine.step()` as the per-iteration engine heartbeat
- [x] Walk through `LLMEngine.generate()` loop to connect scheduler + model runner

---

## Phase 2: PagedAttention Memory Management
**Goal:** Master the mapping between logical blocks and physical blocks.

**Files:**
- `nanovllm/engine/block_manager.py`
- `nanovllm/engine/sequence.py`

- [x] Read `BlockManager.allocate()` and `may_append()`
- [x] Understand free/used block pools (`free_block_ids`, `used_block_ids`)
- [x] Understand `block_table`, `num_cached_tokens`, `num_blocks` in `Sequence`
- [x] Understand prefix caching via block hash chaining

---

## Phase 3: Attention & KV Cache Kernels
**Goal:** Understand how attention computes over non-contiguous KV cache blocks.

**Files:**
- `nanovllm/layers/attention.py`
- `nanovllm/engine/model_runner.py`

- [x] Read `store_kvcache_kernel` and `store_kvcache` in `attention.py`
- [x] Understand prefill vs decode paths in `Attention.forward`
- [x] Understand how `slot_mapping` and `block_tables` are built in `prepare_prefill` / `prepare_decode`
- [x] Understand `seqlen_q` vs `seqlen_k` and why they differ with prefix cache
- [ ] Run `python example.py` on GPU to verify FlashAttention paths execute correctly
- [ ] Run `python bench.py` and compare throughput with/without `enforce_eager`

---

## Phase 4: Scheduling & Preemption
**Goal:** Understand continuous batching and preemption under memory pressure.

**Files:**
- `nanovllm/engine/scheduler.py`

- [x] Study `schedule()` — prefill-first, then decode scheduling
- [x] Inspect `preempt()` and conditions that trigger it
- [x] Follow `postprocess()` completion logic (`eos` and `max_tokens`)

---

## Phase 5: CUDA Graphs & Model Runner
**Goal:** Understand how CUDA graphs accelerate decode and how the model runner orchestrates execution.

**Files:**
- `nanovllm/engine/model_runner.py`

- [x] Understand `model_runner.call()` — single GPU vs multi-GPU (shared memory + events)
- [x] Understand `run_model()` — eager vs CUDA graph path
- [x] Understand `capture_cudagraph()` — why multiple batch sizes, why reversed, graph pool sharing
- [x] Understand `prepare_sample()` and `Sampler.forward()` — temperature scaling + Gumbel-max sampling

---

## Phase 6: The Layers Folder
**Goal:** Understand the custom layer implementations that make inference efficient — KV cache writes, tensor parallelism, positional encoding.

**Reading order (most to least important):**

- `nanovllm/layers/attention.py`
- `nanovllm/layers/sampler.py`
- `nanovllm/layers/linear.py`
- `nanovllm/layers/embed_head.py`
- `nanovllm/layers/rotary_embedding.py`
- `nanovllm/layers/layernorm.py`
- `nanovllm/layers/activation.py`

**Why custom layers?** Standard HuggingFace layers don't know about paged KV cache, tensor parallelism, or packed weights. These replacements have the same math but are KV-cache-aware + tensor-parallel + use fused kernels.

- [x] `attention.py` — `store_kvcache_kernel` (Triton), `store_kvcache`, `Attention.forward` (prefill vs decode FlashAttention paths)
- [x] `sampler.py` — temperature scaling + Gumbel-max sampling (already understood)
- [x] `linear.py` — `ColumnParallelLinear` vs `RowParallelLinear`, weight sharding, all-reduce, packed weight loading
- [x] `embed_head.py` — vocab table split across GPUs, `ParallelLMHead`, `last_indices` trick
- [x] `rotary_embedding.py` — RoPE with explicit `positions` tensor (needed for decode where position ≠ 0)
- [x] `layernorm.py` — fused RMS norm, residual stream pattern
- [x] `activation.py` — fused SiLU for MLP

---

## Phase 7: Live Debugging (GPU)
**Goal:** Prove understanding by instrumenting and observing live behavior.

**Files:**
- `example.py`
- `bench.py`

- [ ] Run `python example.py` end to end
- [ ] Inject print/log lines for free-block counts in `BlockManager` during live run
- [ ] Capture one full iteration trace: `step()` → `schedule()` → `run()` → `postprocess()`
- [ ] Run `python bench.py` and compare throughput with/without `enforce_eager`

---

## Quick Commands
```bash
# activate environment
conda activate nano-vllm

# run demo
python example.py

# run benchmark
python bench.py
```
