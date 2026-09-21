import unittest

from run_titration import normalize_confirmation


class ConfirmationTests(unittest.TestCase):
    def test_accepts_terminal_whitespace_case_and_format_characters(self):
        for value in (
            "EXECUTE",
            "  EXECUTE   ",
            "execute",
            "ＥＸＥＣＵＴＥ",
            "\u200bEXECUTE\ufeff",
        ):
            with self.subTest(value=value):
                self.assertEqual(normalize_confirmation(value), "EXECUTE")

    def test_does_not_accept_partial_confirmation(self):
        self.assertNotEqual(normalize_confirmation("EXEC"), "EXECUTE")


if __name__ == "__main__":
    unittest.main()
