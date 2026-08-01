"""c2 -- automatic prefix caching: exact-prefix reuse over shared blocks."""
from vllm_replica.c1_paged_kv_cache import PagedKVCache
from vllm_replica.c2_prefix_cache import PrefixCache


def test_prefix_cache_reuse_and_pinning():
    BS = 4
    cache = PagedKVCache(n_layer=1, n_head=2, d_head=3, block_size=BS, num_blocks=32)
    prefix = PrefixCache(block_size=BS)
    shared = list(range(12))            # a 12-token "system prompt" = 3 full blocks

    # --- request A: allocate blocks, fill them (pretend), publish to the cache ---
    cache.ensure_capacity(seq_id=0, total_length=len(shared))
    a_blocks = list(cache.block_tables[0])
    prefix.register(shared, cache.block_tables[0], cache)   # publishes + pins 3 blocks
    print("A registered blocks:", a_blocks)

    # --- request B: same prefix + its own tail; should reuse A's 3 blocks ---
    promptB = shared + [99, 98, 97]
    n_cached = prefix.attach_prefix(seq_id=1, token_ids=promptB, cache=cache)
    assert n_cached == 12                                   # skipped all 3 shared blocks
    assert cache.block_tables[1] == a_blocks                # SAME physical blocks reused
    print(f"B reused {n_cached} cached tokens -> blocks {cache.block_tables[1]}")

    # --- a request that shares tokens but NOT from position 0 reuses nothing ---
    n_bad = prefix.attach_prefix(seq_id=2, token_ids=[1, 2, 3] + shared, cache=cache)
    assert n_bad == 0
    print("non-position-0 overlap reused 0 tokens (APC's exact-prefix limit)")

    # --- A finishes: its blocks are pinned by the cache, so they SURVIVE ---
    free_before = cache.allocator.num_free
    cache.free(seq_id=0)                      # A done; refcount drops but pin keeps them
    assert cache.allocator.num_free == free_before   # nothing returned to the pool
    # ...and B (still holding the shared blocks) can keep using them
    k_still_there = cache.block_tables[1]
    assert k_still_there == a_blocks
    print("A finished but pinned prefix blocks survived — B still holds them")

    print("\nc2 OK — exact-prefix reuse, contiguous-from-0 limit, pinned survival")
