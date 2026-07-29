"""Build one deterministic generated certificate for an external verifier."""

from __future__ import annotations

import argparse
from pathlib import Path

from metamath_generator.compose import compose
from metamath_generator.database import TheoremDatabase
from metamath_generator.export import export_metamath
from metamath_generator.parser import parse


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()

    database = parse(args.source)
    store = TheoremDatabase.from_parsed(database)
    theorem = compose(
        database.statements["ax-mp"],
        [None, store.get_by_name("ax-1")],
        name="external-smoke",
        database=database,
    )
    theorem_id, added = store.add(theorem)
    if not added:
        raise RuntimeError("external smoke theorem was unexpectedly deduplicated")
    theorem = store[theorem_id]

    fragment = args.destination.with_suffix(".fragment.mm")
    export_metamath(theorem, store, fragment)
    args.destination.write_text(
        args.source.read_text(encoding="utf-8")
        + "\n"
        + fragment.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    fragment.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
