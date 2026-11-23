import unittest
from unittest.mock import patch

from run_titration import normalize_confirmation
from run_three_stage_demo import _auto_confirm, parser as three_stage_parser


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

    def test_three_stage_yes_enables_noninteractive_mode(self):
        args = three_stage_parser().parse_args(["--observe", "0", "--execute", "--stage", "all", "--yes"])
        self.assertTrue(args.yes)

    def test_auto_confirm_never_reads_stdin(self):
        with patch("builtins.input", side_effect=AssertionError("stdin must not be read")):
            _auto_confirm("checkpoint", "continue")


if __name__ == "__main__":
    unittest.main()
