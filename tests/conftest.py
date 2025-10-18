"""Test configuration ensuring the repository root is importable.

Pytest invokes test modules via its own entry-point script which lives
outside of this repository.  When run that way the interpreter does not
automatically prepend the project root to ``sys.path``.  The harness code
and the lightweight ``numpy`` replacement both live at the repository
root, so we explicitly add the directory to ``sys.path`` here before any
tests import project modules.
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
