# Day 02: `LMCache is unhealthy`, a config schema that silently ignored every key

Setting up disaggregated prefill/decode (vLLM `0.26.0` + LMCache `0.5.2`, NIXL transport), both
servers started fine, the model loaded, NIXL initialised, and yet every request still ran locally.
The cause was upstream of the transport: **LMCache accepted a config file whose keys it no longer
recognised, warned once at INFO level, and came up with peer-to-peer disaggregation disabled.**

## The symptom

Startup logged, buried in hundreds of lines:

```
Unknown configuration key: enable_nixl
Unknown configuration key: nixl_role
Unknown configuration key: nixl_peer_host
```

and then, in the config dump it prints on every engine init:

```
'enable_pd': False
```

Nothing failed. The server served. On the first real request it flipped to:

```
LMCache WARNING: LMCache is unhealthy, skipping store operation
LMCache INFO: Reqid: ..., Total tokens 2007, LMCache hit tokens: 0, need to load: 0
```

`hit tokens: 0` with a healthy-looking HTTP 200 and a perfectly coherent completion. The decoder
had simply recomputed the prompt itself.

## Why the config was wrong

The YAML came from vLLM's own example directory
(`examples/disaggregated/lmcache/disagg_prefill_lmcache_v1/`), pulled at the matching `v0.26.0`
tag. The example still used LMCache's **pre-PD schema**. The installed LMCache had renamed the
whole block, and its config loader treats unknown keys as warnings rather than errors.

Rename map, read out of the installed `lmcache/v1/config.py`, not from docs:

| Old key | New key |
|---|---|
| `enable_nixl: True` | `enable_pd: True` |
| `nixl_role: "sender" / "receiver"` | `pd_role: "sender" / "receiver"` |
| `nixl_peer_host` | `pd_peer_host` |
| `nixl_peer_port` (single) | `pd_peer_init_port` + `pd_peer_alloc_port` + `pd_peer_query_port` |
| *(implied by `enable_nixl`)* | `transfer_channel: "nixl"` |

Plus validation the new schema enforces: `pd_role`, `pd_buffer_size` and `pd_buffer_device` are
**required** whenever `enable_pd: True`, and PD mode auto-sets `save_unfull_chunk=True` (it says so
in the log: *"PD requires save_unfull_chunk=True for complete KV cache transfer"*).

## The working config

Sender (prefiller, GPU 0) has a `pd_peer_host`, so it **dials**:

```yaml
chunk_size: 256
local_cpu: False
max_local_cpu_size: 0

enable_pd: True
transfer_channel: "nixl"
pd_role: "sender"
pd_peer_host: "localhost"
pd_peer_init_port: 7300
pd_peer_alloc_port: 7301
pd_peer_query_port: 7302
pd_buffer_size: 1073741824      # 1 GiB
pd_buffer_device: "cuda"
pd_backend_mode: "async"
pd_skip_proxy_notification: True
```

Receiver (decoder, GPU 1) is identical minus `pd_peer_host`, so it **binds** those three ports.

Confirmation that the keys landed: the config dump on the sender now reads
`'enable_pd': True, 'pd_role': 'sender', 'pd_peer_init_port': [7300], ...` and there are **no
`Unknown configuration key` lines**. That "no warnings" check is the only reliable signal: the
config object is normalised into lists and defaults, so eyeballing the YAML proves nothing.

## Environment coupling that isn't in the YAML

Three settings live outside the config file and break the handoff just as silently:

- **`PYTHONHASHSEED` must be identical on both servers.** The decoder locates staged KV by hashing
  chunk keys; unequal hash seeds mean the lookup misses and you get `hit tokens: 0` with a
  *healthy* engine: the same symptom, a completely different cause.
- **`UCX_TLS=cuda_ipc,cuda_copy,tcp`**, from vLLM's launcher script, not the LMCache docs.
- **Distinct `lmcache_rpc_port` per role** (`producer1` / `consumer1`) in
  `kv_connector_extra_config`; these name ZMQ socket paths, and a collision hangs init.

## An open lead

The receiver's log contains, twice:

```
(EngineCore pid=3450)     assert config.pd_peer_host is not None
```

The decoder config deliberately omits `pd_peer_host` (that's what makes it the listener), yet a
code path in the receiver asserts on it. This is a **lead, not a verified fix**. It lines up with
the handshake never completing on a single node, but I haven't confirmed which call site raises it
or whether it's fatal to that path. Worth pinning down before assuming the two-node deployment is
the only route.

## Transferable lessons

- **A config loader that warns on unknown keys is a trap.** Every setting was ignored, the process
  stayed up, and the failure surfaced one layer down as "unhealthy". Grep startup logs for
  `Unknown config` before trusting any handoff.
- **Read the schema out of the installed library, not the example repo.** The example was pulled at
  the exact matching vLLM tag and was still wrong; vLLM and LMCache version independently.
- **Trust the logs, not the output text.** Every failure mode here returned a fluent, correct-looking
  completion, because falling back to local recompute is the *designed* degradation.

See also: [`../results/d2-disagg-prefill-decode.md`](../results/d2-disagg-prefill-decode.md),
[`../disagg-prefill-decode/disaggregated-prefill-decode-vllm-lmcache.ipynb`](../disagg-prefill-decode/disaggregated-prefill-decode-vllm-lmcache.ipynb).
