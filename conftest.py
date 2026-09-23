"""Put the repository root on sys.path so tests can import `src` and `tools`.

Without this, pytest prepends the test file's package root -- tests/ has no
__init__.py, so that is tests/utils -- and every `from src...` fails. Running
`python -m pytest` also works, but only if everyone remembers the `-m`.

Running a test file directly still fails: conftest.py is not loaded then.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
