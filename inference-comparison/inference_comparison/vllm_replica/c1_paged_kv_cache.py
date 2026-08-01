"""
vLLM-replica / component 1 -- PagedAttention KV cache  (no sharing yet)
======================================================================
The KV cache is split into fixed-size BLOCKS held in one pre-allocated pool.
Each sequence owns a BLOCK TABLE mapping its logical blocks to physical blocks,
which need not be contiguous -- so a sequence grows one block at a time and
freed blocks are instantly reusable.

Sharing between sequences (prefix caching) is NOT enabled yet, so each physical
block has exactly one owner and reference counting is unnecessary. We will add
it back in component 2 when blocks start being shared.

Pool layout: blocks[num_blocks, n_layer, 2(k/v), block_size, n_head, d_head]
"""
from __future__ import annotations
import torch


class BlockAllocator:
    """Tracks which physical block ids are free, via a simple free list."""

    def __init__(self, num_blocks: int):
        """Create a pool of `num_blocks` blocks, all initially free."""
        self.num_blocks = num_blocks
        self._free: list[int] = list(range(num_blocks))

    def allocate(self) -> int:
        """Hand out one free block id (raises if the pool is exhausted)."""
        if not self._free:
            raise MemoryError("KV cache out of blocks (raise num_blocks)")
        return self._free.pop()

    def free(self, block_id: int) -> None:
        """Return a block id to the pool so it can be handed out again."""
        self._free.append(block_id)

    @property
    def num_free(self) -> int:
        """How many blocks are currently available."""
        return len(self._free)


class PagedKVCache:
    """Stores KV at logical token positions via a per-sequence paged block table."""

    K_INDEX, V_INDEX = 0, 1          # dim 2 of the pool: 0 = Key, 1 = Value

    # ---- 1. set-up ------------------------------------------------------- #
    def __init__(self, n_layer: int, n_head: int, d_head: int,
                 block_size: int = 8, num_blocks: int = 512):
        """Allocate the pool tensor and the per-sequence bookkeeping.

        n_layer/n_head/d_head : model geometry (sizes the pool).
        block_size            : tokens per block.
        num_blocks            : total blocks that physically exist.
        """
        self.n_layer, self.n_head, self.d_head = n_layer, n_head, d_head
        self.block_size = block_size
        self.allocator = BlockAllocator(num_blocks)

        # one contiguous pool holding every block's KV, for every layer
        self.blocks = torch.zeros(num_blocks, n_layer, 2, block_size,
                                  n_head, d_head)

        # seq_id -> [physical block ids]; list position = logical block index
        self.block_tables: dict[int, list[int]] = {}
        # physical id -> how many owners
        self.ref_count: dict[int, int] = {}

    # ---- 2. grow capacity as the sequence lengthens ---------------------- #
    def _blocks_needed(self, total_length: int) -> int:
        """Ceiling division: blocks required to hold `total_length` tokens."""
        return (total_length + self.block_size - 1) // self.block_size

    def ensure_capacity(self, seq_id: int, total_length: int) -> None:
        """Grow this sequence's block table until it can address `total_length`
        tokens. Idempotent: a no-op if the table is already large enough."""
        block_table = self.block_tables.setdefault(seq_id, [])
        while len(block_table) < self._blocks_needed(total_length):
            physical_block = self.allocator.allocate()
            self.ref_count[physical_block] = self.ref_count.get(physical_block, 0) + 1
            block_table.append(physical_block)

    # ---- 3. write new tokens' KV into blocks ----------------------------- #
    def _locate(self, block_table: list[int],
                position: int) -> tuple[int, int]:
        """Translate an absolute token position into (physical_block, slot)."""
        return block_table[position // self.block_size], position % self.block_size

    def write(self, seq_id: int, layer_idx: int, first_position: int,
              keys: torch.Tensor, values: torch.Tensor) -> None:
        """Scatter K/V for new tokens into blocks.

        keys, values : shape [num_new_tokens, n_head, d_head]. Token i is stored
        at absolute position `first_position + i`.
        """
        block_table = self.block_tables[seq_id]
        for token_offset in range(keys.shape[0]):
            absolute_position = first_position + token_offset
            physical_block, slot = self._locate(block_table, absolute_position)
            self.blocks[physical_block, layer_idx, self.K_INDEX, slot] = keys[token_offset]
            self.blocks[physical_block, layer_idx, self.V_INDEX, slot] = values[token_offset]

    # ---- 4. read the whole history back ---------------------------------- #
    def read(self, seq_id: int, layer_idx: int, total_length: int
             ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather K/V for positions 0..total_length-1 into two [L, H, D] tensors.

        The scattered physical blocks are collapsed into dense, position-ordered
        tensors via one advanced-indexing gather -- attention never sees the paging.
        """
        block_table = self.block_tables[seq_id]
        physical_blocks = torch.tensor(
            [block_table[position // self.block_size]
             for position in range(total_length)]
        )
        slots = torch.arange(total_length) % self.block_size
        keys = self.blocks[physical_blocks, layer_idx, self.K_INDEX, slots]
        values = self.blocks[physical_blocks, layer_idx, self.V_INDEX, slots]
        return keys, values

    # ---- 5. release the sequence's blocks when it finishes --------------- #
    def free(self, seq_id: int) -> None:
        """Return all of a sequence's blocks to the pool. With no sharing, every
        block has one owner, so each goes straight back. Safe to double-call."""
        for physical_block in self.block_tables.pop(seq_id, []):
            self.ref_count[physical_block] -= 1
            if self.ref_count[physical_block] == 0:
                del self.ref_count[physical_block]
                self.allocator.free(physical_block)

    # ---- diagnostics ----------------------------------------------------- #
    def utilization(self) -> float:
        """Fraction of the pool currently in use (0.0-1.0); diagnostic only."""
        used = self.allocator.num_blocks - self.allocator.num_free
        return used / self.allocator.num_blocks


    def pin(self, physical_block: int) -> None:
        """Add an extra reference so a block outlives its producing sequence
        (the prefix cache uses this to keep completed prefix blocks alive)."""

        self.ref_count[physical_block] = self.ref_count.get(physical_block, 0) + 1

    def adopt_shared_blocks(self, seq_id: int, physical_block_ids: list[int]) -> None:
        """Prepend already-populated physical blocks onto this sequence's (empty)
        table, sharing them (ref-counted, not copied)."""

        block_table = self.block_tables.setdefault(seq_id, [])
        assert not block_table, "adopt shared blocks before allocating fresh ones"
        for physical_block in physical_block_ids:
            self.ref_count[physical_block] += 1
            block_table.append(physical_block)
