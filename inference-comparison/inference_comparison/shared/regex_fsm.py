"""regex -> NFA (Thompson) -> DFA over the byte alphabet (0..255).

Shared by both structured-decoding components, the same way vLLM and SGLang both
sit on a grammar backend like xgrammar or outlines and differ only in how they
drive the automaton.

Supported syntax: literals, '.', char classes [a-z0-9_], shortcuts \\d \\w \\s,
quantifiers * + ?, alternation |, groups ( ).

The DFA exposes what a constrained decoder needs:
    allowed_bytes(state) -> set[int]
    is_accepting(state)  -> bool
    step(state, byte)    -> state | None
"""
from __future__ import annotations

FULL = frozenset(range(256))
DIGIT = frozenset(range(ord("0"), ord("9") + 1))
WORD = (frozenset(range(ord("a"), ord("z") + 1))
        | frozenset(range(ord("A"), ord("Z") + 1))
        | DIGIT | {ord("_")})
SPACE = frozenset(ord(c) for c in " \t\n\r")


class _NFA:
    def __init__(self):
        self.trans: dict[int, list[tuple[frozenset | None, int]]] = {}
        self.n = 0

    def new_state(self) -> int:
        s = self.n
        self.n += 1
        self.trans[s] = []
        return s

    def add(self, src: int, symbols, dst: int) -> None:
        # symbols is None for epsilon, otherwise a frozenset of byte values
        self.trans[src].append((symbols, dst))


class _Frag:
    __slots__ = ("start", "accept")

    def __init__(self, start: int, accept: int):
        self.start, self.accept = start, accept


class _Parser:
    def __init__(self, pattern: str, nfa: _NFA):
        self.p = pattern
        self.i = 0
        self.nfa = nfa

    def peek(self):
        return self.p[self.i] if self.i < len(self.p) else None

    def next(self):
        c = self.p[self.i]
        self.i += 1
        return c

    def parse(self) -> _Frag:
        frag = self.alt()
        if self.i != len(self.p):
            raise ValueError(f"unexpected char at {self.i}: {self.p[self.i:]!r}")
        return frag

    def alt(self) -> _Frag:
        frags = [self.concat()]
        while self.peek() == "|":
            self.next()
            frags.append(self.concat())
        if len(frags) == 1:
            return frags[0]
        s, a = self.nfa.new_state(), self.nfa.new_state()
        for f in frags:
            self.nfa.add(s, None, f.start)
            self.nfa.add(f.accept, None, a)
        return _Frag(s, a)

    def concat(self) -> _Frag:
        frags = []
        while self.peek() not in (None, "|", ")"):
            frags.append(self.repeat())
        if not frags:
            s = self.nfa.new_state()          # empty match
            return _Frag(s, s)
        for i in range(len(frags) - 1):
            self.nfa.add(frags[i].accept, None, frags[i + 1].start)
        return _Frag(frags[0].start, frags[-1].accept)

    def repeat(self) -> _Frag:
        f = self.atom()
        q = self.peek()
        if q in ("*", "+", "?"):
            self.next()
            s, a = self.nfa.new_state(), self.nfa.new_state()
            if q == "*":
                self.nfa.add(s, None, f.start); self.nfa.add(s, None, a)
                self.nfa.add(f.accept, None, f.start); self.nfa.add(f.accept, None, a)
            elif q == "+":
                self.nfa.add(s, None, f.start)
                self.nfa.add(f.accept, None, f.start); self.nfa.add(f.accept, None, a)
            else:  # ?
                self.nfa.add(s, None, f.start); self.nfa.add(s, None, a)
                self.nfa.add(f.accept, None, a)
            return _Frag(s, a)
        return f

    def _lit(self, symbols: frozenset) -> _Frag:
        s, a = self.nfa.new_state(), self.nfa.new_state()
        self.nfa.add(s, symbols, a)
        return _Frag(s, a)

    def atom(self) -> _Frag:
        c = self.peek()
        if c == "(":
            self.next()
            f = self.alt()
            assert self.next() == ")", "unbalanced ')'"
            return f
        if c == "[":
            return self._lit(self._char_class())
        if c == ".":
            self.next()
            return self._lit(FULL)
        if c == "\\":
            self.next()
            return self._lit(self._escape(self.next()))
        return self._lit(frozenset({ord(self.next())}))

    def _escape(self, e: str) -> frozenset:
        return {"d": DIGIT, "w": WORD, "s": SPACE}.get(e, frozenset({ord(e)}))

    def _char_class(self) -> frozenset:
        assert self.next() == "["
        negate = False
        if self.peek() == "^":
            self.next(); negate = True
        out: set[int] = set()
        while self.peek() != "]":
            ch = self.next()
            if ch == "\\":
                out |= set(self._escape(self.next()))
                continue
            if self.peek() == "-" and self.i + 1 < len(self.p) and self.p[self.i + 1] != "]":
                self.next()                       # consume '-'
                hi = self.next()
                out |= set(range(ord(ch), ord(hi) + 1))
            else:
                out.add(ord(ch))
        self.next()                               # consume ']'
        fs = frozenset(out)
        return FULL - fs if negate else fs


class RegexFSM:
    def __init__(self, pattern: str):
        nfa = _NFA()
        frag = _Parser(pattern, nfa).parse()
        self._nfa = nfa
        self._accept_nfa = frag.accept
        self._build_dfa(frag.start)

    def _eps_closure(self, states: frozenset) -> frozenset:
        stack, seen = list(states), set(states)
        while stack:
            s = stack.pop()
            for sym, dst in self._nfa.trans[s]:
                if sym is None and dst not in seen:
                    seen.add(dst); stack.append(dst)
        return frozenset(seen)

    def _build_dfa(self, nfa_start: int) -> None:
        start = self._eps_closure(frozenset({nfa_start}))
        self.trans: list[dict[int, int]] = []
        self.accepting: set[int] = set()
        index: dict[frozenset, int] = {start: 0}
        order: list[frozenset] = [start]
        self.trans.append({})
        i = 0
        while i < len(order):
            cur = order[i]
            if self._accept_nfa in cur:
                self.accepting.add(i)
            # group by byte, collecting the reachable NFA states
            for b in range(256):
                move = set()
                for s in cur:
                    for sym, dst in self._nfa.trans[s]:
                        if sym is not None and b in sym:
                            move.add(dst)
                if not move:
                    continue
                tgt = self._eps_closure(frozenset(move))
                if tgt not in index:
                    index[tgt] = len(order)
                    order.append(tgt)
                    self.trans.append({})
                self.trans[i][b] = index[tgt]
            i += 1
        self.start = 0

    def allowed_bytes(self, state: int) -> set[int]:
        return set(self.trans[state].keys())

    def is_accepting(self, state: int) -> bool:
        return state in self.accepting

    def step(self, state: int, byte: int):
        return self.trans[state].get(byte)
