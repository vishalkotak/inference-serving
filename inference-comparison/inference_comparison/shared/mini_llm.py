"""
shared/mini_llm.py
==================
A *tiny*, randomly-initialised decoder-only Transformer used by BOTH the
vLLM-replica and the SGLang-replica.

There is deliberately **no model download and no external tokenizer**:
    * weights are random (torch default init) + a fixed seed for reproducibility,
    * the tokenizer is byte-level (vocab = 256, id == byte value).

The generated *text* is therefore meaningless. That is fine: these projects
replicate the *serving-engine mechanisms*, not model quality. Every mechanism
is validated on its own terms (see each demo's asserts).

The one important design choice: attention is delegated to a pluggable
`AttentionBackend`. The SAME `TinyLLM` runs under vLLM's PagedAttention backend
and under SGLang's RadixAttention backend -- exactly like a real engine swapping
its KV-cache/attention backend under a fixed model.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Protocol, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Tokenizer (byte-level, no files)                                            #
# --------------------------------------------------------------------------- #
class ByteTokenizer:
    """Maps a str <-> list[int] using raw UTF-8 bytes. vocab_size == 256.

    Token id 0 doubles as our EOS marker in demos (the NUL byte is never
    produced from normal text, so it is safe to reserve)."""
    vocab_size = 256
    eos_id = 0

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: list[int]) -> str:
        return bytes(i for i in ids if i != self.eos_id).decode("utf-8", "replace")


# --------------------------------------------------------------------------- #
# Config                                                                       #
# --------------------------------------------------------------------------- #
@dataclass
class TinyLLMConfig:
    vocab_size: int = 256
    n_layer: int = 2
    n_head: int = 4
    d_head: int = 16
    d_ff: int = 256
    max_seq: int = 1024
    seed: int = 0

    @property
    def d_model(self) -> int:
        return self.n_head * self.d_head


# --------------------------------------------------------------------------- #
# Attention backend protocol                                                   #
# --------------------------------------------------------------------------- #
@dataclass
class SeqContext:
    """Per-sequence bookkeeping passed into the model on every forward step.

    seq_id       : unique id of the running sequence
    start_pos    : absolute position of the FIRST token in this step's input
                   (== number of tokens already committed to the KV cache)
    """
    seq_id: int
    start_pos: int
    extra: dict = field(default_factory=dict)


class AttentionBackend(Protocol):
    """A backend owns the KV cache and computes attention.

    Contract for `attention`:
      * inputs q,k,v have shape [T, n_head, d_head]  (T = tokens in this step)
      * the backend must (1) append k,v for these T tokens to its cache for
        `ctx.seq_id` at `layer_idx`, then (2) return attention output of shape
        [T, d_model], where each of the T query rows attends causally over the
        full history (cached prefix + the new tokens up to and including itself).
    """
    def attention(self, layer_idx: int, q: torch.Tensor, k: torch.Tensor,
                  v: torch.Tensor, ctx: SeqContext) -> torch.Tensor: ...


# --------------------------------------------------------------------------- #
# The shared attention math (used by every backend after it gathers K/V)       #
# --------------------------------------------------------------------------- #
def causal_attention(q: torch.Tensor, k_all: torch.Tensor, v_all: torch.Tensor,
                     n_prefix: int) -> torch.Tensor:
    """Multi-head causal attention.

    q      : [T, H, D]   queries for the T new tokens
    k_all  : [P+T, H, D] keys for prefix (P) + new (T) tokens
    v_all  : [P+T, H, D]
    n_prefix (P): number of cached prefix tokens (query i may see all P of them
                  plus new tokens j <= i).

    Returns [T, H*D].
    """
    T, H, D = q.shape
    L = k_all.shape[0]
    P = n_prefix
    assert L == P + T, f"expected {P}+{T} keys, got {L}"

    # [H, T, D] x [H, D, L] -> [H, T, L]
    qh = q.permute(1, 0, 2)                      # [H, T, D]
    kh = k_all.permute(1, 2, 0)                  # [H, D, L]
    scores = torch.matmul(qh, kh) / math.sqrt(D) # [H, T, L]

    # causal mask: new query i (abs pos P+i) may attend to key at abs pos <= P+i
    key_pos = torch.arange(L)                    # 0..L-1  == absolute positions
    qry_pos = torch.arange(P, P + T).unsqueeze(1)  # [T,1]
    allowed = key_pos.unsqueeze(0) <= qry_pos    # [T, L]
    scores = scores.masked_fill(~allowed.unsqueeze(0), float("-inf"))

    probs = torch.softmax(scores, dim=-1)        # [H, T, L]
    vh = v_all.permute(1, 0, 2)                  # [H, L, D]
    out = torch.matmul(probs, vh)                # [H, T, D]
    return out.permute(1, 0, 2).reshape(T, H * D)


# --------------------------------------------------------------------------- #
# Model                                                                        #
# --------------------------------------------------------------------------- #
class TinyBlock(nn.Module):
    def __init__(self, cfg: TinyLLMConfig):
        super().__init__()
        d = cfg.d_model
        self.n_head, self.d_head = cfg.n_head, cfg.d_head
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, cfg.d_ff), nn.GELU(),
                                 nn.Linear(cfg.d_ff, d))

    def forward(self, layer_idx, h, backend: AttentionBackend, ctx: SeqContext):
        T = h.shape[0]
        x = self.ln1(h)
        q, k, v = self.qkv(x).split(h.shape[-1], dim=-1)
        shp = (T, self.n_head, self.d_head)
        q, k, v = q.view(shp), k.view(shp), v.view(shp)
        attn = backend.attention(layer_idx, q, k, v, ctx)  # [T, d_model]
        h = h + self.proj(attn)
        h = h + self.mlp(self.ln2(h))
        return h


class TinyLLM(nn.Module):
    def __init__(self, cfg: TinyLLMConfig):
        super().__init__()
        torch.manual_seed(cfg.seed)
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq, cfg.d_model)
        self.layers = nn.ModuleList([TinyBlock(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.eval()  # inference-only replicas

    @torch.no_grad()
    def forward(self, input_ids: list[int], backend: AttentionBackend,
                ctx: SeqContext) -> torch.Tensor:
        """Run one step (prefill: T>1, or decode: T==1). Returns logits for the
        LAST position only -- that is all a serving engine samples from."""
        ids = torch.tensor(input_ids, dtype=torch.long)
        pos = torch.arange(ctx.start_pos, ctx.start_pos + len(input_ids))
        h = self.tok_emb(ids) + self.pos_emb(pos)
        for i, layer in enumerate(self.layers):
            h = layer(i, h, backend, ctx)
        h = self.ln_f(h)
        return self.lm_head(h[-1])   # [vocab]


# --------------------------------------------------------------------------- #
# Sampling helpers                                                             #
# --------------------------------------------------------------------------- #
def greedy(logits: torch.Tensor) -> int:
    return int(torch.argmax(logits).item())


def sample(logits: torch.Tensor, temperature: float = 1.0,
           generator: Optional[torch.Generator] = None) -> int:
    if temperature <= 0:
        return greedy(logits)
    probs = torch.softmax(logits / temperature, dim=-1)
    return int(torch.multinomial(probs, 1, generator=generator).item())


# --------------------------------------------------------------------------- #
# A trivial reference backend (no paging / no radix) used only to validate     #
# that the fancy backends produce identical numbers.                           #
# --------------------------------------------------------------------------- #
class DenseBackend:
    """Stores KV in plain per-(seq,layer) python lists. Ground truth."""
    def __init__(self):
        self.k: dict = {}   # (seq,layer) -> Tensor [N,H,D]
        self.v: dict = {}

    def _key(self, seq, layer):
        return (seq, layer)

    def attention(self, layer_idx, q, k, v, ctx: SeqContext):
        key = self._key(ctx.seq_id, layer_idx)
        pk, pv = self.k.get(key), self.v.get(key)
        n_prefix = 0 if pk is None else pk.shape[0]
        k_all = k if pk is None else torch.cat([pk, k], 0)
        v_all = v if pv is None else torch.cat([pv, v], 0)
        self.k[key], self.v[key] = k_all, v_all
        return causal_attention(q, k_all, v_all, n_prefix)

    def free(self, seq_id):
        for layer in list({l for (s, l) in self.k if s == seq_id}):
            self.k.pop((seq_id, layer), None)
            self.v.pop((seq_id, layer), None)
