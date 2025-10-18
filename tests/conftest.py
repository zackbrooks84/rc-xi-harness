"""Test configuration utilities for the RC+xi harness suite."""
from __future__ import annotations

import sys
from pathlib import Path


def _ensure_repo_root_on_path() -> None:
    """Insert the repository root at the front of ``sys.path``.

    Pytest's console script entry point can launch from the site-packages
    directory, leaving the project root absent from ``sys.path``. Importing
    the ``harness`` package (which implements the public test harness) then
    fails during collection. By inserting the root explicitly we guarantee
    the source package is importable regardless of the invocation style.
    """

    repo_root = Path(__file__).resolve().parent.parent
    root_str = str(repo_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


_ensure_repo_root_on_path()
