import importlib
import os
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import nllb_test


class PhrasesTest(unittest.TestCase):
    def test_phrases_are_20_unique_non_empty_strings(self):
        self.assertEqual(len(nllb_test.PHRASES), 20)
        self.assertTrue(all(isinstance(p, str) and p.strip() for p in nllb_test.PHRASES))
        self.assertEqual(len(set(nllb_test.PHRASES)), 20)


class LoadPhrasesTest(unittest.TestCase):
    def _write(self, content):
        f = tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False)
        f.write(content)
        f.close()
        self.addCleanup(os.remove, f.name)
        return f.name

    def test_without_argument_returns_default(self):
        with mock.patch.object(sys, "argv", ["nllb_test.py"]):
            self.assertEqual(nllb_test.load_phrases(), nllb_test.PHRASES)

    def test_file_is_stripped_and_blank_lines_ignored(self):
        path = self._write("  Où est l'été ?  \n\n\t\nÇa va.\n")
        with mock.patch.object(sys, "argv", ["nllb_test.py", path]):
            self.assertEqual(nllb_test.load_phrases(), ["Où est l'été ?", "Ça va."])

    def test_empty_file_falls_back_to_default(self):
        path = self._write("\n  \n\n")
        with mock.patch.object(sys, "argv", ["nllb_test.py", path]):
            self.assertEqual(nllb_test.load_phrases(), nllb_test.PHRASES)


class ModelEnvTest(unittest.TestCase):
    def tearDown(self):
        importlib.reload(nllb_test)

    def test_default_model(self):
        with mock.patch.dict(os.environ):
            os.environ.pop("NLLB_MODEL", None)
            importlib.reload(nllb_test)
            self.assertEqual(nllb_test.MODEL, "facebook/nllb-200-distilled-600M")

    def test_model_from_environment(self):
        with mock.patch.dict(os.environ, {"NLLB_MODEL": "autre/modele"}):
            importlib.reload(nllb_test)
            self.assertEqual(nllb_test.MODEL, "autre/modele")


class MainTest(unittest.TestCase):
    def test_exits_when_kab_latn_is_missing(self):
        tokenizer = mock.Mock(unk_token_id=3)
        tokenizer.convert_tokens_to_ids.return_value = 3
        transformers = mock.Mock()
        transformers.AutoTokenizer.from_pretrained.return_value = tokenizer
        modules = {"torch": mock.Mock(), "transformers": transformers}
        with mock.patch.dict(sys.modules, modules), \
                mock.patch.object(sys, "argv", ["nllb_test.py"]):
            with self.assertRaises(SystemExit) as ctx:
                nllb_test.main()
        self.assertIn("kab_Latn", str(ctx.exception.code))


if __name__ == "__main__":
    unittest.main()
