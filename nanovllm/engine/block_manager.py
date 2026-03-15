from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0       # number of sequences sharing this block (>1 = prefix sharing)
        self.hash = -1           # hash of token content, -1 means block is partial (not yet full)
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    # Hash a block's token content, chained with the previous block's hash.
    # Chaining means block i's hash depends on all tokens before it — so two blocks
    # with identical content but different prefixes get different hashes.
    # This is what makes prefix caching safe and correct.
    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int) -> Block:
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    # Called by scheduler before scheduling a prefill — checks if there are
    # enough free blocks to hold the entire sequence.
    def can_allocate(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= seq.num_blocks

    # Allocate KV cache blocks for a new sequence (prefill phase).
    # Implements prefix caching: if a block's hash matches an existing cached block,
    # reuse it instead of allocating fresh. Once a cache miss occurs, all subsequent
    # blocks must be fresh (prefix must be contiguous from the start).
    def allocate(self, seq: Sequence):
        assert not seq.block_table
        h = -1
        cache_miss = False
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            # only hash full blocks — partial last block can't be shared
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True
            if cache_miss:
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:
                seq.num_cached_tokens += self.block_size  # this block's K/V already in GPU memory
                if block_id in self.used_block_ids:
                    block = self.blocks[block_id]
                    block.ref_count += 1   # another sequence now shares this block
                else:
                    block = self._allocate_block(block_id) # block was previously used, then deallocated, but its hash entry was never removed from hash_to_block_id
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            seq.block_table.append(block_id)

    # Free all blocks when a sequence finishes.
    # Shared prefix blocks (ref_count > 1) are only returned to the free pool
    # once no sequence references them anymore.
    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    # Called by scheduler before each decode step — checks if a new block needs
    # to be allocated. A new block is only needed when the sequence just filled
    # its last block (len % block_size == 1 means first token of a new block).
    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    # Called after a token is appended to the sequence during decode.
    # Handles two block boundary events:
    #   - Sequence just entered a new block (% == 1): allocate a fresh block
    #   - Sequence just completed a block (% == 0): finalize it with a hash
    #     so future sequences can reuse it as a cached prefix
    def may_append(self, seq: Sequence):
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        if len(seq) % self.block_size == 1:
            assert last_block.hash != -1   # previous block must be finalized before we move on
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
        elif len(seq) % self.block_size == 0:
            # block just became full — hash it and register for prefix sharing
            assert last_block.hash == -1
            token_ids = seq.block(seq.num_blocks-1)
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1 # using the previous block's hash to calculate the current block's prefix for chaining
            h = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id
        else:
            assert last_block.hash == -1   # mid-block: still partial, nothing to do
