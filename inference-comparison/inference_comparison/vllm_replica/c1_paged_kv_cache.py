"""Paged KV cache: fixed-size blocks in one pool, one block table per sequence.

Blocks don't have to be contiguous, so a sequence grows a block at a time and freed
blocks are reusable immediately. Blocks are refcounted because c2 lets two sequences
point at the same physical block.

Pool layout: blocks[num_blocks, n_layer, 2(k/v), block_size, n_head, d_head]
"""
from __future__ import annotations
import torch


class BlockAllocator:
    """Free list over physical block ids."""

    def __init__(self, num_blocks: int):
        self.num_blocks = num_blocks
        self._free: list[int] = list(range(num_blocks))

    def allocate(self) -> int:
        if not self._free:
            raise MemoryError("KV cache out of blocks (raise num_blocks)")
        return self._free.pop()

    def free(self, block_id: int) -> None:
        self._free.append(block_id)

    @property
    def num_free(self) -> int:
        return len(self._free)


class PagedKVCache:
    """Stores KV at logical token positions via a per-sequence paged block table."""

    K_INDEX, V_INDEX = 0, 1

    def __init__(self, n_layer: int, n_head: int, d_head: int,
                 block_size: int = 8, num_blocks: int = 512):
        self.n_layer, self.n_head, self.d_head = n_layer, n_head, d_head
        self.block_size = block_size
        self.allocator = BlockAllocator(num_blocks)
        self.blocks = torch.zeros(num_blocks, n_layer, 2, block_size,
                                  n_head, d_head)
        # seq_id -> [physical block ids]; list position is the logical block index
        self.block_tables: dict[int, list[int]] = {}
        self.ref_count: dict[int, int] = {}

    def _blocks_needed(self, total_length: int) -> int:
        return (total_length + self.block_size - 1) // self.block_size

    def ensure_capacity(self, seq_id: int, total_length: int) -> None:
        """Grow the block table until it covers total_length tokens. Idempotent."""
        block_table = self.block_tables.setdefault(seq_id, [])
        while len(block_table) < self._blocks_needed(total_length):
            physical_block = self.allocator.allocate()
            self.ref_count[physical_block] = self.ref_count.get(physical_block, 0) + 1
            block_table.append(physical_block)

    def _locate(self, block_table: list[int],
                position: int) -> tuple[int, int]:
        return block_table[position // self.block_size], position % self.block_size

    def write(self, seq_id: int, layer_idx: int, first_position: int,
              keys: torch.Tensor, values: torch.Tensor) -> None:
        """Scatter [num_new_tokens, n_head, d_head] K/V starting at first_position."""
        block_table = self.block_tables[seq_id]
        for token_offset in range(keys.shape[0]):
            absolute_position = first_position + token_offset
            physical_block, slot = self._locate(block_table, absolute_position)
            self.blocks[physical_block, layer_idx, self.K_INDEX, slot] = keys[token_offset]
            self.blocks[physical_block, layer_idx, self.V_INDEX, slot] = values[token_offset]

    def read(self, seq_id: int, layer_idx: int, total_length: int
             ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather positions 0..total_length-1 into two [L, H, D] tensors."""
        block_table = self.block_tables[seq_id]
        physical_blocks = torch.tensor(
            [block_table[position // self.block_size]
             for position in range(total_length)]
        )
        slots = torch.arange(total_length) % self.block_size
        keys = self.blocks[physical_blocks, layer_idx, self.K_INDEX, slots]
        values = self.blocks[physical_blocks, layer_idx, self.V_INDEX, slots]
        return keys, values

    def free(self, seq_id: int) -> None:
        """Drop one reference to each of the sequence's blocks. Safe to call twice."""
        for physical_block in self.block_tables.pop(seq_id, []):
            self.ref_count[physical_block] -= 1
            if self.ref_count[physical_block] == 0:
                del self.ref_count[physical_block]
                self.allocator.free(physical_block)

    def utilization(self) -> float:
        used = self.allocator.num_blocks - self.allocator.num_free
        return used / self.allocator.num_blocks

    def pin(self, physical_block: int) -> None:
        """Hold an extra reference so a block outlives the sequence that filled it."""
        self.ref_count[physical_block] = self.ref_count.get(physical_block, 0) + 1

    def adopt_shared_blocks(self, seq_id: int, physical_block_ids: list[int]) -> None:
        """Point seq_id at already-filled blocks instead of copying them."""
        block_table = self.block_tables.setdefault(seq_id, [])
        assert not block_table, "adopt shared blocks before allocating fresh ones"
        for physical_block in physical_block_ids:
            self.ref_count[physical_block] += 1
            block_table.append(physical_block)
