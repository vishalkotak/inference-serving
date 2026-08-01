"""Prefix caching: reuse KV blocks across requests that start with the same tokens.

This is only a lookup table from prefix tokens to physical block id. The sharing
itself happens in c1 via adopt_shared_blocks/pin. Sequence ids don't matter here;
all this records is which blocks hold which tokens.
"""
from __future__ import annotations

from vllm_replica.c1_paged_kv_cache import PagedKVCache


class PrefixCache:

    def __init__(self, block_size: int):
        self.block_size = block_size
        # Keyed on the cumulative prefix rather than just this block's tokens, so a
        # hit implies every earlier token matched too. With block_size 4 and tokens
        # [10,11,12,13, 20,21,22,23]:
        #   block 0 -> (10,11,12,13)
        #   block 1 -> (10,11,12,13, 20,21,22,23)
        self._prefix_to_block: dict[tuple, int] = {}

    def _num_full_blocks(self, token_ids: list[int]) -> int:
        # A trailing partial block isn't cacheable; more tokens can still land in it.
        return len(token_ids) // self.block_size

    def match(self, token_ids: list[int]) -> tuple[int, list[int]]:
        """Return (num_cached_tokens, reusable_physical_block_ids)."""
        reusable_blocks: list[int] = []
        for block_index in range(self._num_full_blocks(token_ids)):
            cumulative_key = tuple(token_ids[: (block_index + 1) * self.block_size])
            physical_block = self._prefix_to_block.get(cumulative_key)
            if physical_block is None:
                break                    # prefixes diverge here, so does everything after
            reusable_blocks.append(physical_block)
        return len(reusable_blocks) * self.block_size, reusable_blocks

    def register(self, token_ids: list[int], block_table: list[int], cache: PagedKVCache | None = None) -> None:
        """Publish a sequence's full blocks for reuse, pinning them so they survive it."""
        for block_index in range(self._num_full_blocks(token_ids)):
            cumulative_key = tuple(token_ids[: (block_index + 1) * self.block_size])
            if cumulative_key not in self._prefix_to_block:
                self._prefix_to_block[cumulative_key] = block_table[block_index]
                if cache is not None:
                    cache.pin(block_table[block_index])

    def attach_prefix(self, seq_id: int, token_ids: list[int], cache: PagedKVCache) -> int:
        """Point seq_id at any reusable prefix blocks; return how many tokens that covers."""
        num_cached_tokens, reusable_blocks = self.match(token_ids)
        if reusable_blocks:
            cache.adopt_shared_blocks(seq_id, reusable_blocks)
        return num_cached_tokens
