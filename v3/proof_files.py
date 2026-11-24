"""Episode-local symbolic files over one initialized v2.1 Interaction."""

import copy
import json

from metamath_generator.parser import MetamathParser
from pa_prover_v21.interaction import compact, parse_command
from v3.tokenization.pa_symbols import FILE_NAMES, FILE_OPS, PASymbolTokenizer


class FileBudgetExceeded(ValueError):
    pass


class ProofFiles:
    """Route proof commands through advance() so speculative writes follow the proof branch."""

    def __init__(self, flow, *, max_calls=64, max_io_tokens=8192):
        if not flow.events or flow.pending is not None:
            raise ValueError("attach files to an initialized interaction outside a trial")
        if any(type(b) is not int or b < 0 for b in (max_calls, max_io_tokens)):
            raise ValueError("file budgets must be nonnegative integers")
        self.flow = flow
        self.tokenizer = PASymbolTokenizer(flow.session.b.db)
        self.calls_left = max_calls
        self.tokens_left = max_io_tokens
        self._pages = {"FILE_PLAN": (), "FILE_SCRATCH": ()}
        self._parent_pages = None
        self._decisions = [event["content"]["op"] for event in flow.events
                           if event["role"] == "model" and event["content"]["op"] in {"COMMIT", "BACKTRACK"}]
        seen = dict.fromkeys(r["state_key"] for r in flow.memory)
        self._state_ids = {key: i for i, key in enumerate(seen)}
        target = flow.session.target
        self._problem = [
            {"goal": str(target.conclusion)},
            *({"hypothesis": str(h.expr)} for h in target.hypotheses),
            {"types": dict(target.variable_types)},
            {"distinct": sorted(tuple(sorted(pair)) for pair in target.d_constraints)},
        ]

    def advance(self, command, backend, config, remaining):
        """Use the existing executor for every logical step, including commit and restore."""
        op, _ = parse_command(command)
        before = self._pages
        child = self.flow.fork()
        try:
            child.execute(command, backend, config, remaining)
        except Exception:
            # Keep failed branches and any proof cache with branch-local names out of the parent.
            backend.session, backend.cache = self.flow.session, {}
            raise
        self.flow = child
        if op == "TRY":
            self._parent_pages = before
            key = child.memory[-1]["state_key"]
            self._state_ids.setdefault(key, len(self._state_ids))
        elif op in {"COMMIT", "BACKTRACK"}:
            self._decisions.append(op)
            if op == "BACKTRACK":
                self._pages = self._parent_pages
                backend.session, backend.cache = child.session, {}
            self._parent_pages = None

    def _charge(self, text):
        cost = len(self.tokenizer.encode(text))
        if cost > self.tokens_left:
            raise FileBudgetExceeded("file token budget exhausted")
        self.tokens_left -= cost

    @staticmethod
    def _index(text):
        if not text.isascii() or not text.isdecimal():
            raise ValueError("expected a nonnegative decimal record index or count")
        return int(text)

    def _read(self, name, offset, count):
        if not 1 <= count <= 8:
            raise ValueError("read between 1 and 8 records")
        if name == "FILE_PROBLEM":
            rows = self._problem
        elif name == "FILE_LIBRARY":
            rows = self.flow.events[-1]["content"]["state"]["library"]
        elif name == "FILE_TRIALS":
            rows = self.flow.memory
        else:
            rows = self._pages[name]
        if offset > len(rows):
            raise ValueError("read offset past end of file")
        page = rows[offset:offset + count]
        if name == "FILE_TRIALS":
            page = [{"state": self._state_ids[r["state_key"]], "action": json.loads(r["action"]),
                     "outcome": r["outcome"], "closed_count": len(r["closed"]),
                     "decision": self._decisions[i] if i < len(self._decisions) else None}
                    for i, r in enumerate(page, start=offset)]
        elif name == "FILE_LIBRARY":
            page = [{"id": r["id"], "statement": r["statement"]} for r in page]
        end = offset + len(page)
        return {"rows": copy.deepcopy(list(page)), "next": end if end < len(rows) else None}

    def _record(self, name, words):
        allowed = ("wff",) if name == "FILE_PLAN" else ("wff", "term")
        if len(words) < 2 or words[0] not in allowed:
            raise ValueError(f"{name} expects a nonempty {' or '.join(allowed)} expression")
        if len(self.tokenizer.encode(" ".join(words))) > 256:
            raise FileBudgetExceeded("file record exceeds 256 tokens")
        # Use the grammar from the source but only variables in the task or current public state.
        parser = MetamathParser()
        parser.database = copy.copy(self.flow.session.b.db)
        visible_types = self.flow.events[-1]["content"]["state"]["types"]
        parser.database.variable_types = {**self.flow.session.target.variable_types, **visible_types}
        return str(parser.parse_expression(words))

    def file(self, command):
        """Execute one model command. The returned data belongs to an ENGINE response."""
        if self.calls_left == 0:
            raise FileBudgetExceeded("file call budget exhausted")
        self.calls_left -= 1
        self._charge(command)
        words = command.split()
        if len(words) < 2 or len(command.splitlines()) != 1:
            raise ValueError("expected a single file command")
        op, name, *args = words
        if op not in FILE_OPS or name not in FILE_NAMES:
            raise ValueError("unknown file operation or file name")
        new_pages = self._pages
        if op == "FREAD":
            if len(args) != 2:
                raise ValueError("FREAD requires an offset and a count")
            reply = self._read(name, self._index(args[0]), self._index(args[1]))
        else:
            if name not in self._pages:
                raise ValueError(f"{name} is written by the engine")
            rows = self._pages[name]
            if op == "FAPPEND":
                if len(rows) >= 64:
                    raise FileBudgetExceeded("file is full")
                index, payload = len(rows), args
            else:
                if not args:
                    raise ValueError("FWRITE requires an existing record index")
                index, payload = self._index(args[0]), args[1:]
                if index >= len(rows):
                    raise ValueError("FWRITE requires an existing record index")
            record = self._record(name, payload)
            new_pages = {**self._pages, name: rows[:index] + (record,) + rows[index + 1:]}
            reply = {"index": index}
        self._charge(compact(reply))
        self._pages = new_pages
        return reply
