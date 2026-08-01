# check_fsm.py  — run: poetry run python check_fsm.py
from shared.regex_fsm import RegexFSM

def matches(fsm, s):
    st = fsm.start
    for b in s.encode():
        st = fsm.step(st, b)
        if st is None:
            return False
    return fsm.is_accepting(st)

f = RegexFSM(r'[0-9]+')
assert matches(f, "12345") and not matches(f, "12a") and not matches(f, "")
assert not f.is_accepting(f.start)          # need at least one digit

json = RegexFSM(r'\{"age": [0-9]+\}')
assert matches(json, '{"age": 42}')
# the whole prefix is forced: at the start ONLY '{' is legal
assert json.allowed_bytes(json.start) == {ord("{")}

print("regex_fsm OK — JSON pattern has", len(json.trans), "DFA states")
