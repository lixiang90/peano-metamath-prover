"""Fixed UTF-8 bytes: no corpus-dependent vocabulary or token-ID migration."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable


PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
MEM_ID = 3
BYTE_OFFSET = 4
VOCAB_SIZE = 260


class ByteTokenizer:
    pad_id = PAD_ID
    bos_id = BOS_ID
    eos_id = EOS_ID
    mem_id = MEM_ID
    byte_offset = BYTE_OFFSET
    vocab_size = VOCAB_SIZE

    def __len__(self) -> int:
        return self.vocab_size

    def encode(self, text: str, *, bos: bool = False, eos: bool = False) -> list[int]:
        result = [byte + self.byte_offset for byte in text.encode("utf-8")]
        if bos:
            result.insert(0, self.bos_id)
        if eos:
            result.append(self.eos_id)
        return result

    def decode(self, ids: Iterable[int]) -> str:
        values: list[int] = []
        for token in ids:
            token = int(token)
            if not 0 <= token < self.vocab_size:
                raise ValueError(f"token ID outside byte vocabulary: {token}")
            if token >= self.byte_offset:
                values.append(token - self.byte_offset)
        # A generated prefix can end inside a multibyte character. Replacement
        # decoding makes it inspectable; complete UTF-8 encodings round-trip.
        return bytes(values).decode("utf-8", errors="replace")

    def specification(self) -> dict[str, int | str]:
        return {
            "name": "utf8-bytes-v1", "encoding": "utf-8",
            "pad_id": self.pad_id, "bos_id": self.bos_id,
            "eos_id": self.eos_id, "mem_id": self.mem_id,
            "byte_offset": self.byte_offset, "vocab_size": self.vocab_size,
        }

    def fingerprint(self) -> str:
        serialized = json.dumps(self.specification(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


UTF8ByteTokenizer = ByteTokenizer
