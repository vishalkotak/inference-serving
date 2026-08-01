"""
vLLM-replica / component 2 -- automatic prefix caching
======================================================
GOAL: many requests share an identical opening (a system prompt). Computing the
KV for those identical tokens over and over is pure waste. Instead: after one
request fills the KV for a prefix, REMEMBER which physical blocks hold it, keyed
by the exact tokens. The next request with the same opening REUSES those blocks
and skips straight to its unique part.

This class is ONLY a lookup table (prefix tokens -> physical block id). The
actual block sharing is done by c1's adopt_shared_blocks + pin.

Important points: Why does this class not care about sequence identifiers.
It's only job is the token→block mapping. It takes "here are the tokens, here are the physical blocks that
hold their KV" and records that. Whether those blocks came from sequence 0, sequence 5, or a hand-built
test list is irrelevant
"""
from __future__ import annotations

from vllm_replica.c1_paged_kv_cache import PagedKVCache



class PrefixCache:

    def __init__(self, block_size: int):
        self.block_size = block_size

        # key   = tuple(prompt_ids[:(block_index+1)*block_size])
        #         i.e. ALL tokens up to and including this block (cumulative prefix)
        # value = physical block id in the PagedKVCache pool
        # Using the cumulative prefix (not just this block's tokens) means a match
        # is a match only if EVERY earlier token is identical too -> collision-free.
        self._prefix_to_block: dict[tuple, int] = {}

        # Concrete example, block_size = 4, tokens [10,11,12,13, 20,21,22,23]:
        # block 0 key = (10,11,12,13)
        # block 1 key = (10,11,12,13, 20,21,22,23) ← includes block 0's tokens too

    def _num_full_blocks(self, token_ids: list[int]) -> int:
        """How many COMPLETELY-filled blocks these tokens make (trailing partial
        block is not cacheable — its contents aren't fixed yet)."""

        return len(token_ids) // self.block_size


    def match(self, token_ids: list[int]) -> tuple[int, list[int]]:
        """Return (num_cached_tokens, reusable_physical_block_ids).
        Walk full blocks from the front; STOP at the first miss — a prefix match
        must be contiguous from position 0 (no skipping gaps)."""

        reusable_blocks: list[int] = []
        for block_index in range(self._num_full_blocks(token_ids)):
            cumulative_key = tuple(token_ids[: (block_index + 1) * self.block_size])
            physical_block = self._prefix_to_block.get(cumulative_key)
            if physical_block is None: # first mismatch = prefixes diverge here
                break                 # everything after is necessarily new too
            reusable_blocks.append(physical_block)
        return len(reusable_blocks) * self.block_size, reusable_blocks


    def register(self, token_ids: list[int], block_table: list[int], cache: PagedKVCache | None = None) -> None:
        """Publish a filled sequence's full blocks for future reuse, pinning each
        in the cache so it outlives the producing sequence."""

        for block_index in range(self._num_full_blocks(token_ids)):
            cumulative_key = tuple(token_ids[: (block_index + 1) * self.block_size])
            if cumulative_key not in self._prefix_to_block:
                self._prefix_to_block[cumulative_key] = block_table[block_index]
                if cache is not None:
                    cache.pin(block_table[block_index])

    def attach_prefix(self, seq_id: int, token_ids: list[int], cache: PagedKVCache) -> int:
        """Adopt any reusable prefix blocks into seq_id's empty block table and
        return the number of already-cached tokens (prefill can skip them)."""

        num_cached_tokens, reusable_blocks = self.match(token_ids)
        if reusable_blocks:
            cache.adopt_shared_blocks(seq_id, reusable_blocks)
        return num_cached_tokens
