import contextlib
import io
import runpy
import sys
import unittest
from unittest.mock import patch


class DotNxsModuleCliTests(unittest.TestCase):
    def test_python_m_dotnxs_invokes_cli(self):
        output = io.StringIO()
        with patch.object(sys, "argv", ["dotnxs", "--help"]), contextlib.redirect_stdout(output):
            with self.assertRaises(SystemExit) as raised:
                runpy.run_module("dotnxs", run_name="__main__")
        self.assertEqual(raised.exception.code, 0)
        self.assertIn("Create and inspect single-file DOTNXS archives", output.getvalue())


if __name__ == "__main__":
    unittest.main()
