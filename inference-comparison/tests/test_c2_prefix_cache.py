from vllm_replica.c1_paged_kv_cache import PagedKVCache
from vllm_replica.c2_prefix_cache import PrefixCache


def test_prefix_cache_reuse_and_pinning():
    BS = 4
    cache = PagedKVCache(n_layer=1, n_head=2, d_head=3, block_size=BS, num_blocks=32)
    prefix = PrefixCache(block_size=BS)
    shared = list(range(12))            # 12-token system prompt, 3 full blocks

    cache.ensure_capacity(seq_id=0, total_length=len(shared))
    a_blocks = list(cache.block_tables[0])
    prefix.register(shared, cache.block_tables[0], cache)

    # same opening plus its own tail, so it should land on A's blocks
    promptB = shared + [99, 98, 97]
    n_cached = prefix.attach_prefix(seq_id=1, token_ids=promptB, cache=cache)
    assert n_cached == 12
    assert cache.block_tables[1] == a_blocks

    # overlap that doesn't start at position 0 is not reusable
    n_bad = prefix.attach_prefix(seq_id=2, token_ids=[1, 2, 3] + shared, cache=cache)
    assert n_bad == 0

    # A finishes, but the pin keeps its blocks out of the free list so B keeps them
    free_before = cache.allocator.num_free
    cache.free(seq_id=0)
    assert cache.allocator.num_free == free_before
    assert cache.block_tables[1] == a_blocks
