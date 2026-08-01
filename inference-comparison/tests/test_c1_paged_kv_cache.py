import torch

from vllm_replica.c1_paged_kv_cache import PagedKVCache


def test_paged_kv_cache_roundtrip():
    BLOCK_SIZE = 4
    cache = PagedKVCache(n_layer=1, n_head=2, d_head=3,
                         block_size=BLOCK_SIZE, num_blocks=8)
    SEQ = 0

    cache.ensure_capacity(SEQ, total_length=6)
    assert len(cache.block_tables[SEQ]) == 2

    prompt_keys = torch.randn(6, 2, 3)
    prompt_values = torch.randn(6, 2, 3)
    cache.write(SEQ, layer_idx=0, first_position=0,
                keys=prompt_keys, values=prompt_values)

    # decoding past position 7 pulls in a third block
    cache.ensure_capacity(SEQ, total_length=9)
    assert len(cache.block_tables[SEQ]) == 3

    extra_keys = torch.randn(3, 2, 3)
    extra_values = torch.randn(3, 2, 3)
    cache.write(SEQ, layer_idx=0, first_position=6,
                keys=extra_keys, values=extra_values)

    k_back, v_back = cache.read(SEQ, layer_idx=0, total_length=9)
    assert k_back.shape == (9, 2, 3)
    assert torch.allclose(k_back[:6], prompt_keys)
    assert torch.allclose(k_back[6:], extra_keys)
    assert torch.allclose(v_back[:6], prompt_values)
    assert torch.allclose(v_back[6:], extra_values)

    free_before = cache.allocator.num_free
    cache.free(SEQ)
    assert cache.allocator.num_free == free_before + 3
