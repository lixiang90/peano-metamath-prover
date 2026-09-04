"""Keep unified entry points and bilingual instructions from drifting apart."""

from __future__ import annotations

from collections import Counter
import importlib.util
from pathlib import Path
import re
import shlex
import unittest


ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS = [ROOT / "README.md", ROOT / "README-zh.md", ROOT / "CONTRIBUTING.md"] + sorted(
    (ROOT / "docs").glob("*.md")
)
LINK = re.compile(r"\]\(([^)]+)\)")
BLOCK = re.compile(r"```(bash|powershell)\n(.*?)```", re.DOTALL)


class DocumentationTests(unittest.TestCase):
    def test_local_links_resolve(self):
        for path in DOCUMENTS:
            for link in LINK.findall(path.read_text(encoding="utf-8")):
                if "://" in link or link.startswith("#"):
                    continue
                target = link.split("#", 1)[0]
                with self.subTest(document=path.name, target=target):
                    self.assertTrue((path.parent / target).exists())

    def test_bilingual_readmes_have_matching_structure_and_commands(self):
        english = (ROOT / "README.md").read_text(encoding="utf-8")
        chinese = (ROOT / "README-zh.md").read_text(encoding="utf-8")
        self.assertEqual(BLOCK.findall(english), BLOCK.findall(chinese))
        self.assertEqual(
            re.findall(r"^(#+) ", english, re.MULTILINE),
            re.findall(r"^(#+) ", chinese, re.MULTILINE),
        )
        def destinations(content):
            return Counter(link for link in LINK.findall(content) if "img.shields.io" not in link)

        self.assertEqual(destinations(english), destinations(chinese))

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch needed to load HTPS CLI")
    def test_current_workflow_cli_examples_parse_without_running(self):
        from metamath_generator.cli import build_parser as generator_parser
        from neural_prover.cli import build_parser as neural_parser
        from htps_prover.cli import build_parser as htps_parser

        parsers = {
            "metamath_generator": generator_parser(),
            "neural_prover": neural_parser(),
            "htps_prover": htps_parser(),
        }
        count = 0
        for path in [ROOT / "README.md", ROOT / "docs/training-and-evaluation.md", ROOT / "docs/pa-plus-htps-training.md"]:
            for language, block in BLOCK.findall(path.read_text(encoding="utf-8")):
                if language != "bash":
                    continue
                for line in re.sub(r"\\\s*\n", " ", block).splitlines():
                    args = shlex.split(line, comments=True)
                    if len(args) < 3 or args[:2] != ["python", "-m"]:
                        continue
                    parser = parsers.get(args[2])
                    if parser is None:
                        continue
                    with self.subTest(document=path.name, command=line):
                        parser.parse_args(args[3:])
                        count += 1
        self.assertGreaterEqual(count, 15)


if __name__ == "__main__":
    unittest.main()
