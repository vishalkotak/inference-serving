"""c1 -- paged KV cache: grow, scatter-write, gather-read, free."""
import torch

from vllm_replica.c1_paged_kv_cache import PagedKVCache


def test_paged_kv_cache_roundtrip():
    BLOCK_SIZE = 4
    cache = PagedKVCache(n_layer=1, n_head=2, d_head=3,
                         block_size=BLOCK_SIZE, num_blocks=8)
    SEQ = 0
    print("start        :", cache.allocator.num_free, "free blocks")
    cache.ensure_capacity(SEQ, total_length=6)
    assert len(cache.block_tables[SEQ]) == 2
    print("after prefill:", cache.allocator.num_free, "free |",
          "block table =", cache.block_tables[SEQ])

    prompt_keys   = torch.randn(6, 2, 3)      # [tokens, n_head, d_head]
    prompt_values = torch.randn(6, 2, 3)
    cache.write(SEQ, layer_idx=0, first_position=0,
                keys=prompt_keys, values=prompt_values)

    # --- stage 2 again: decode until we cross into a THIRD block ---
    cache.ensure_capacity(SEQ, total_length=9)
    assert len(cache.block_tables[SEQ]) == 3        # NOW a third block is pulled in
    # write tokens 7 and 8 (positions 6, 7 already fit; here we add positions 6..8)
    extra_keys   = torch.randn(3, 2, 3)             # positions 6, 7, 8
    extra_values = torch.randn(3, 2, 3)
    cache.write(SEQ, layer_idx=0, first_position=6,
                keys=extra_keys, values=extra_values)
    print("after decode :", cache.allocator.num_free, "free |",
          "block table =", cache.block_tables[SEQ])

    # --- stage 4: read the whole 9-token history back ---
    k_back, v_back = cache.read(SEQ, layer_idx=0, total_length=9)
    assert k_back.shape == (9, 2, 3)
    assert torch.allclose(k_back[:6], prompt_keys)    # prompt part
    assert torch.allclose(k_back[6:], extra_keys)     # decoded tokens
    assert torch.allclose(v_back[:6], prompt_values)
    assert torch.allclose(v_back[6:], extra_values)
    print("read-back    : 9 tokens gathered, KV matches exactly")

    # --- stage 5: free -> all 3 blocks return ---
    free_before = cache.allocator.num_free
    cache.free(SEQ)
    assert cache.allocator.num_free == free_before + 3
