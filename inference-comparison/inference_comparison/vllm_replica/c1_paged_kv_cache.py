# GOAL: for every token a sequence has seen, remember its Key and Value vectors
# (one pair per layer per head) so future tokens can attend to them.
#
# NAIVE VERSION (the thing we're trying to beat):
#   kv[seq_id] = a big tensor sized to MAX_LEN, allocated up front.
# PAIN: we don't know how long each sequence gets, so we reserve for the worst
# case. A 20-token chat reserves space for 2000 tokens -> ~99% wasted.
#
# FIX (the OS "paging" idea): chop storage into small fixed-size BLOCKS in one
# shared pool. Give each sequence a little lookup table mapping its logical
# blocks to whatever physical blocks are free. Grow one block at a time.

import torch

class BlockAllocator:
    """Tracks which physical block ids are used. That's literally all it does."""

    def __init__(self, num_blocks: int):
        self.num_blocks = num_blocks
        self._free = list(range(num_blocks))

    def allocate(self) -> int:
        if not self._free:
            raise MemoryError("Out of blocks")
        return self._free.pop()

    def free(self, block_id: int) -> None:
        self._free.append(block_id)


# I need to look up KV by: (which block, which layer, K-or-V, which slot in block,
#                           which head, the vector itself)
num_blocks = 512
n_layer = 2
n_head = 4
d_head = 16
block_size = 8

# WHY store all layers in one block? Because then a sequence
# needs exactly ONE block table for the whole model, not one per layer. That's
# what real vLLM does, and it keeps the bookkeeping simple.
blocks = torch.zeros(
        num_blocks,
        n_layer,
        2,
        block_size,
        n_head,
        d_head)


# seq_id -> [physical block ids].  Position in the list = logical block index.
#   block_tables[7] == [37, 4, 91]  means seq 7's logical blocks 0,1,2 live in
#   physical blocks 37, 4, 91 (non-adjacent — that's fine, that's the point).
block_tables: dict[int, list[int]] = {}



