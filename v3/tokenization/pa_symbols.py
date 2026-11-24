"""Lossless PA+ symbols and v2.1 protocol pieces, with the v2 byte alphabet as fallback."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable

from metamath_generator.model import Database
from pa_prover_v2.tokenizer import BOS_ID, BYTE_OFFSET, EOS_ID, MEM_ID, PAD_ID, VOCAB_SIZE, ByteTokenizer


# Closed fields and action names of the high-level proof interface, independent of a corpus.
FIELDS = """
action active assumptions attempts budget closed content depth distinct event events execution
exhausted facts formula generalize goal goals history id index kind lemma library library_snapshot
memory mode op pending_trial protocol role search_status seconds state statement status trials
types variable
""".split()
WORDS = """
APPLY BACKTRACK COMMIT HAVE INDUCT READ STRENGTHEN TRY UNFOLD USE
STATE ACTION ENGINE MODEL OPEN ROOT_READY UNKNOWN UNKNOWN_WITHIN_BUDGET
feedback scratchpad true false null
""".split()
# This extension has a published order. Append later controls after its last ID.
FILE_OPS = ("FREAD", "FWRITE", "FAPPEND")
FILE_NAMES = ("FILE_PROBLEM", "FILE_PLAN", "FILE_SCRATCH", "FILE_TRIALS", "FILE_LIBRARY")
WORD = r"[\w][\w-]*"
# Consecutive formal atoms print with a single space, as in Node.to_prefix().
# JOIN expresses immediate adjacency when the original surface has no space.
JOIN = VOCAB_SIZE
FIRST_ATOM = JOIN + 1


class PASymbolTokenizer:
    pad_id = PAD_ID
    bos_id = BOS_ID
    eos_id = EOS_ID
    mem_id = MEM_ID

    def __init__(self, database: Database, *, file_tokens: bool = True):
        slots = {f"{stem}{i}" for stem in ("v", "x", "zzv") for i in range(32)}
        formal = set(database.symbols) | database.variables | database.logical_assertions.keys() | slots
        protocol = set(WORDS) | {f'"{key}":' for key in FIELDS}
        original = formal | protocol
        self.pieces = tuple(sorted(original))
        self.version = "pa-symbol-v1"
        if file_tokens:
            self.pieces += tuple(p for p in FILE_OPS + FILE_NAMES if p not in original)
            self.version = "pa-symbol-files-v1"
        self.ids = {piece: FIRST_ATOM + i for i, piece in enumerate(self.pieces)}
        self.formal_ids = {self.ids[piece] for piece in formal}
        special = sorted((p for p in self.pieces if not re.fullmatch(WORD, p)), key=lambda s: (-len(s), s))
        self.lexer = re.compile("|".join([*(re.escape(p) for p in special), WORD]))
        self.bytes = ByteTokenizer()

    @property
    def vocab_size(self) -> int:
        return FIRST_ATOM + len(self.pieces)

    def __len__(self) -> int:
        return self.vocab_size

    def fingerprint(self) -> str:
        spec = [self.version, self.bytes.fingerprint(), self.pieces, sorted(self.formal_ids)]
        return hashlib.sha256(json.dumps(spec, ensure_ascii=False).encode()).hexdigest()

    def encode(self, text: str, *, bos: bool = False, eos: bool = False) -> list[int]:
        ids = [BOS_ID] if bos else []
        end = 0
        previous_formal = False
        for match in self.lexer.finditer(text):
            token = self.ids.get(match.group())
            if token is None:
                continue
            gap = text[end:match.start()]
            formal = token in self.formal_ids
            if previous_formal and formal and gap == "":
                ids.append(JOIN)
            elif not (previous_formal and formal and gap == " "):
                ids.extend(self.bytes.encode(gap))
            ids.append(token)
            previous_formal = formal
            end = match.end()
        ids.extend(self.bytes.encode(text[end:]))
        if eos:
            ids.append(EOS_ID)
        return ids

    def encode_segments(self, segments: list[str]) -> list[list[int]]:
        """Keep each token within one prompt, engine, or supervised model span."""
        encoded = []
        previous_formal = False
        for part in segments:
            ids = self.encode(part)
            if ids:
                if previous_formal and ids[0] in self.formal_ids:
                    ids.insert(0, JOIN)
                previous_formal = ids[-1] in self.formal_ids
            encoded.append(ids)
        return encoded

    def decode(self, ids: Iterable[int]) -> str:
        data = bytearray()
        previous_formal = False
        for item in ids:
            token = int(item)
            if FIRST_ATOM <= token < self.vocab_size:
                formal = token in self.formal_ids
                if formal and previous_formal:
                    data.append(32)
                data.extend(self.pieces[token - FIRST_ATOM].encode())
                previous_formal = formal
            elif BYTE_OFFSET <= token < VOCAB_SIZE:
                data.append(token - BYTE_OFFSET)
                previous_formal = False
            elif token in (JOIN, PAD_ID, BOS_ID, EOS_ID, MEM_ID):
                previous_formal = False
            else:
                raise ValueError(f"unknown PA token ID {token}")
        return data.decode("utf-8")
