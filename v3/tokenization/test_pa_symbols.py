from pathlib import Path
import unittest

from metamath_generator.parser import parse
from pa_symbols import FILE_NAMES, FILE_OPS, JOIN, PASymbolTokenizer


class PASymbolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[2]
        cls.db = parse(root / "formal" / "peano-pa-plus.mm")
        cls.tokenizer = PASymbolTokenizer(cls.db)

    def test_file_tokens_append_to_original_vocabulary(self):
        old = PASymbolTokenizer(self.db, file_tokens=False)
        new = self.tokenizer
        self.assertEqual(old.fingerprint(), "d2b6a0f092934495c5075b44ce1050b15b723815f2dff1dcd4127d8ad0312c2c")
        self.assertEqual((len(old), len(new)), (701, 709))
        self.assertNotEqual(new.fingerprint(), old.fingerprint())
        self.assertTrue(all(new.ids[p] == i for p, i in old.ids.items()))
        for i, atom in enumerate(FILE_OPS + FILE_NAMES, start=len(old)):
            self.assertEqual(new.encode(atom), [i])
            self.assertEqual(new.decode([i]), atom)
        self.assertNotIn(new.ids["FILE_PLAN"], new.encode("FILE_PLAN_B"))
        spans = ["MODEL\n", "FREAD FILE_SCRATCH 0 1\n", "ENGINE\n", "wff = + 0 x0 x0\n"]
        encoded = new.encode_segments(spans)
        self.assertEqual([new.decode(part) for part in encoded], spans)
        self.assertEqual(new.decode(i for part in encoded for i in part), "".join(spans))

    def test_formula_and_unseen_names(self):
        tok = self.tokenizer
        formula = "forall x0 exists x1 and < x0 x1 prime x1"
        self.assertEqual(tok.encode(formula), [tok.ids[s] for s in formula.split()])
        self.assertNotEqual(tok.encode("x0"), tok.encode("x1"))
        for text in ("x999", "zzv132", "future-lemma-37", "prefixforall", "新引理🐢"):
            self.assertEqual(tok.encode(text), tok.bytes.encode(text))
            self.assertEqual(tok.decode(tok.encode(text, bos=True, eos=True)), text)

    def test_spacing_and_supervision_boundaries(self):
        tok = self.tokenizer
        self.assertIn(JOIN, tok.encode("x0+x1"))
        texts = [
            "x0+x1", "x0 + x1", "x0  +\tx1\r\n", '$d x0 x1 $.',
            '{"goal":"forall x0 = x0 x1","variable":"zzv123"}\n',
            "ENGINE\n引理α\nMODEL\nINDUCT\n", "",
        ]
        for text in texts:
            self.assertEqual(tok.decode(tok.encode(text)), text)
            for cut in range(len(text) + 1):
                parts = [text[:cut], "", text[cut:]]
                encoded = tok.encode_segments(parts)
                self.assertEqual([tok.decode(ids) for ids in encoded], parts)
                self.assertEqual(tok.decode(i for span in encoded for i in span), text)


if __name__ == "__main__":
    unittest.main()
