"""Unit tests for the minimal ``numpy.load`` implementation."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import pytest


MAGIC = b"\x93NUMPY"
VERSION = (1, 0)


def _prod(shape: Tuple[int, ...]) -> int:
    total = 1
    for dim in shape:
        total *= dim
    return total


def _format_header(shape: Tuple[int, ...], descr: str, fortran_order: bool) -> bytes:
    shape_repr = str(shape if len(shape) != 1 else (shape[0],))
    header = (
        "{'descr': '"
        + descr
        + "', 'fortran_order': "
        + ("True" if fortran_order else "False")
        + ", 'shape': "
        + shape_repr
        + ", }"
    ).encode("latin1")
    header_len = len(header) + 1  # account for trailing newline
    padding = (16 - ((len(MAGIC) + 2 + 2 + header_len) % 16)) % 16
    return header + (b" " * padding) + b"\n"


def _write_npy(
    path: Path,
    shape: Tuple[int, ...],
    values: Sequence[float],
    *,
    descr: str = "<f8",
    fortran_order: bool = False,
) -> None:
    if _prod(shape) != len(values):
        raise ValueError("values do not match requested shape")

    header = _format_header(shape, descr=descr, fortran_order=fortran_order)
    with path.open("wb") as fh:
        fh.write(MAGIC)
        fh.write(bytes(VERSION))
        fh.write(struct.pack("<H", len(header)))
        fh.write(header)
        if values:
            fh.write(struct.pack("<" + "d" * len(values), *values))


def test_load_round_trips_float_matrix(tmp_path: Path) -> None:
    """Behavioral: ``np.load`` reconstructs a 2x3 float matrix."""

    path = tmp_path / "matrix.npy"
    shape = (2, 3)
    values = [float(i) for i in range(6)]
    _write_npy(path, shape, values)

    loaded = np.load(path)

    assert isinstance(loaded, np.ndarray)
    assert loaded.shape == shape
    assert loaded.tolist() == [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
    mean_value = float(np.mean(loaded))
    assert abs(mean_value - (sum(values) / len(values))) < 1e-9


def test_load_rejects_fortran_order(tmp_path: Path) -> None:
    """Invariant: Fortran-ordered payloads are rejected with a clear error."""

    path = tmp_path / "fortran.npy"
    _write_npy(path, (2, 2), [1.0, 2.0, 3.0, 4.0], fortran_order=True)

    with pytest.raises(NotImplementedError):
        np.load(path)
