# quick_check.py  — run: python quick_check.py
import torch
from shared.mini_llm import (TinyLLM, TinyLLMConfig, ByteTokenizer,
                             SeqContext, DenseBackend, greedy)

tok = ByteTokenizer()
assert tok.decode(tok.encode("hello")) == "hello"          # round-trips

cfg = TinyLLMConfig()
model = TinyLLM(cfg)
ids = tok.encode("the quick brown fox")
logits = model.forward(ids, DenseBackend(), SeqContext(seq_id=0, start_pos=0))
assert logits.shape == (256,)                              # one vector, last pos
print("first predicted byte:", greedy(logits), "— mini_llm OK")
