"""Minimal NumPy-compatible subset implemented in pure Python.

This lightweight module provides just enough functionality for the RC+ξ
public harness to run without depending on the real NumPy distribution.
Only a very small surface area is implemented – enough to support the
operations exercised by the repository and its tests.  The goal is
deterministic, dependency-free numeric utilities covering basic array
operations, statistics, and random sampling.

The implementation focuses on correctness and clarity rather than speed.
Arrays are backed by nested Python lists and the supported broadcasting
rules are limited to the cases required by the harness (matching shapes
or singleton expansion along an axis).  This keeps the code compact while
still matching the semantics relied upon by the tests.
"""

from __future__ import annotations

import builtins
import math
import operator
import random as _py_random
from collections import Counter
from typing import Any, Callable, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

Number = Union[int, float]


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float))


def _deep_copy(value: Any) -> Any:
    if isinstance(value, list):
        return [_deep_copy(v) for v in value]
    return value


def _infer_shape(data: Any) -> Tuple[int, ...]:
    if isinstance(data, list):
        if not data:
            return (0,)
        first_shape = _infer_shape(data[0])
        for item in data[1:]:
            if _infer_shape(item) != first_shape:
                raise ValueError("ragged nested sequences are not supported")
        return (len(data),) + first_shape
    return ()


def _prod(values: Sequence[int]) -> int:
    result = 1
    for v in values:
        result *= v
    return result


def _to_nested_list(value: Any) -> Any:
    if isinstance(value, ndarray):
        return _deep_copy(value._data)
    if isinstance(value, (list, tuple)):
        return [_to_nested_list(v) for v in value]
    return value


def _ensure_shape(data: Any, shape: Tuple[int, ...]) -> Any:
    if not shape:
        if isinstance(data, bool):
            return bool(data)
        if isinstance(data, float):
            return float(data)
        if isinstance(data, int):
            return int(data)
        return data
    if not isinstance(data, list) or len(data) != shape[0]:
        raise ValueError("input does not match requested shape")
    return [_ensure_shape(item, shape[1:]) for item in data]


def _wrap(data: Any) -> Any:
    if isinstance(data, list):
        return ndarray(data)
    return data


def _normalize_axis(axis: int, ndim: int) -> int:
    if axis < 0:
        axis += ndim
    if not (0 <= axis < ndim):
        raise ValueError("axis out of range")
    return axis


def _broadcast_shape(shape_a: Tuple[int, ...], shape_b: Tuple[int, ...]) -> Tuple[int, ...]:
    rank = max(len(shape_a), len(shape_b))
    padded_a = (1,) * (rank - len(shape_a)) + shape_a
    padded_b = (1,) * (rank - len(shape_b)) + shape_b
    result: List[int] = []
    for dim_a, dim_b in zip(padded_a, padded_b):
        if dim_a == dim_b:
            result.append(dim_a)
        elif dim_a == 1:
            result.append(dim_b)
        elif dim_b == 1:
            result.append(dim_a)
        else:
            raise ValueError("operands could not be broadcast together")
    return tuple(result)


def _broadcast_to(data: Any, current_shape: Tuple[int, ...], target_shape: Tuple[int, ...]) -> Any:
    if current_shape == target_shape:
        return _deep_copy(data)
    if not target_shape:
        if current_shape:
            raise ValueError("cannot broadcast non-scalar to scalar")
        return _deep_copy(data)
    if not current_shape:
        # Scalar broadcast
        return [_broadcast_to(data, (), target_shape[1:]) for _ in range(target_shape[0])]
    if len(current_shape) != len(target_shape):
        return _broadcast_to(data, (1,) * (len(target_shape) - len(current_shape)) + current_shape, target_shape)
    if current_shape[0] == target_shape[0]:
        return [
            _broadcast_to(item, current_shape[1:], target_shape[1:])
            for item in data
        ]
    if current_shape[0] == 1:
        return [
            _broadcast_to(data[0], current_shape[1:], target_shape[1:])
            for _ in range(target_shape[0])
        ]
    raise ValueError("operands could not be broadcast together")


class ndarray:
    """Pure Python representation of a multidimensional numeric array."""

    _data: Any
    shape: Tuple[int, ...]
    ndim: int
    size: int

    def __init__(self, data: Any):
        nested = _to_nested_list(data)
        shape = _infer_shape(nested)
        self._data = _ensure_shape(nested, shape) if shape else float(nested) if _is_number(nested) else nested
        self.shape = shape
        self.ndim = len(shape)
        self.size = _prod(shape) if shape else 1

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"ndarray(shape={self.shape}, data={self._data})"

    # Iteration ---------------------------------------------------------
    def __iter__(self) -> Iterator[Any]:
        if not self.shape:
            yield self.item()
            return
        for item in self._data:
            yield _wrap(item)

    def __len__(self) -> int:
        if not self.shape:
            raise TypeError("len() of unsized object")
        return self.shape[0]

    def item(self) -> Any:
        if self.shape:
            raise ValueError("item() is only valid for 0-d arrays")
        return self._data

    # Indexing ----------------------------------------------------------
    def __getitem__(self, key: Any) -> Any:
        return _wrap(self._get_item(key))

    def __setitem__(self, key: Any, value: Any) -> None:
        self._set_item(key, value)

    def _normalize_key(self, key: Any) -> Tuple[List[Any], List[int]]:
        if not isinstance(key, tuple):
            key = (key,)
        new_axes: List[int] = []
        processed: List[Any] = []
        for idx in key:
            if idx is None:
                new_axes.append(len(processed))
            else:
                processed.append(idx)
        while len(processed) < self.ndim:
            processed.append(slice(None))
        if len(processed) > self.ndim:
            raise IndexError("too many indices")
        return processed, new_axes

    def _get_item(self, key: Any) -> Any:
        processed, new_axes = self._normalize_key(key)

        def _rec(data: Any, indices: List[Any]) -> Any:
            if not indices:
                return _deep_copy(data)
            idx, *rest = indices
            if isinstance(idx, float):
                if idx.is_integer():
                    idx = int(idx)
                else:
                    raise TypeError("fractional indices are not supported")
            if isinstance(idx, ndarray):
                seq = [int(i) for i in idx.tolist()]
                if not isinstance(data, list):
                    raise IndexError("too many indices")
                return [_rec(data[i], rest) for i in seq]
            if isinstance(idx, slice):
                if not isinstance(data, list):
                    raise IndexError("too many indices")
                rng = range(*idx.indices(len(data)))
                return [_rec(data[i], rest) for i in rng]
            if isinstance(idx, int):
                if not isinstance(data, list):
                    raise IndexError("too many indices")
                return _rec(data[idx], rest)
            raise TypeError("unsupported index type")

        result = _rec(self._data, processed)
        for axis in new_axes:
            result = _expand_axis(result, axis)
        return result

    def _set_item(self, key: Any, value: Any) -> None:
        processed, new_axes = self._normalize_key(key)
        if new_axes:
            raise TypeError("assignment with new axes is not supported")

        def _rec_assign(data: Any, indices: List[Any], val: Any) -> Any:
            if not indices:
                return _to_nested_list(val)
            idx, *rest = indices
            if isinstance(idx, float):
                if idx.is_integer():
                    idx = int(idx)
                else:
                    raise TypeError("fractional indices are not supported")
            if isinstance(idx, slice):
                if not isinstance(data, list):
                    raise IndexError("too many indices")
                rng = range(*idx.indices(len(data)))
                vals = _to_nested_list(val)
                if not isinstance(vals, list) or len(vals) != len(rng):
                    raise ValueError("cannot broadcast assignment value")
                for target, source in zip(rng, vals):
                    data[target] = _rec_assign(data[target], rest, source)
                return data
            if isinstance(idx, ndarray):
                indices = [int(i) for i in idx.tolist()]
                vals = _to_nested_list(val)
                if isinstance(vals, list) and len(vals) == len(indices):
                    for target, source in zip(indices, vals):
                        data[target] = _rec_assign(data[target], rest, source)
                    return data
                for target in indices:
                    data[target] = _rec_assign(data[target], rest, vals)
                return data
            if isinstance(idx, int):
                if not isinstance(data, list):
                    raise IndexError("too many indices")
                data[idx] = _rec_assign(data[idx], rest, val)
                return data
            raise TypeError("unsupported index type")

        self._data = _rec_assign(self._data, processed, value)

    # Convenience -------------------------------------------------------
    def copy(self) -> "ndarray":
        return ndarray(_deep_copy(self._data))

    def tolist(self) -> Any:
        return _deep_copy(self._data)

    def all(self) -> bool:
        return builtins.all(bool(x) for x in _flatten(self))

    @property
    def T(self) -> "ndarray":
        if self.ndim != 2:
            raise ValueError("transpose is only supported for 2D arrays")
        rows, cols = self.shape
        return ndarray([[self._data[r][c] for r in range(rows)] for c in range(cols)])

    def reshape(self, *shape: int) -> "ndarray":
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = tuple(shape[0])  # type: ignore[assignment]
        shape = tuple(shape)
        if not shape:
            raise ValueError("shape must contain at least one dimension")
        known = 1
        unknown = -1
        for dim in shape:
            if dim == -1:
                if unknown != -1:
                    raise ValueError("can only specify one unknown dimension")
                unknown = 1
            else:
                known *= dim
        total = self.size
        if unknown == -1:
            if known != total:
                raise ValueError("cannot reshape array: size mismatch")
            target_shape = shape
        else:
            if total % known:
                raise ValueError("cannot reshape array: size mismatch")
            missing = total // known
            target_shape = tuple(missing if dim == -1 else dim for dim in shape)

        flat = _flatten(self)
        values = flat.copy()

        def _build(dimensions: Tuple[int, ...]) -> Any:
            if not dimensions:
                return values.pop(0)
            count = dimensions[0]
            return [_build(dimensions[1:]) for _ in range(count)]

        data = _build(target_shape)
        return ndarray(data)

    def ravel(self) -> "ndarray":
        return ndarray(_flatten(self))

    # Arithmetic --------------------------------------------------------
    def _binary_op(self, other: Any, op: Callable[[Any, Any], Any]) -> "ndarray":
        arr_other = asarray(other)
        target_shape = _broadcast_shape(self.shape, arr_other.shape)
        a = _broadcast_to(self._data, self.shape, target_shape)
        b = _broadcast_to(arr_other._data, arr_other.shape, target_shape)

        def _rec(x: Any, y: Any) -> Any:
            if isinstance(x, list):
                return [_rec(ix, iy) for ix, iy in zip(x, y)]
            return op(x, y)

        return ndarray(_rec(a, b))

    def _unary_op(self, op: Callable[[Any], Any]) -> "ndarray":
        def _rec(x: Any) -> Any:
            if isinstance(x, list):
                return [_rec(v) for v in x]
            return op(x)

        return ndarray(_rec(self._data))

    def __add__(self, other: Any) -> "ndarray":
        return self._binary_op(other, operator.add)

    def __radd__(self, other: Any) -> "ndarray":
        return self.__add__(other)

    def __sub__(self, other: Any) -> "ndarray":
        return self._binary_op(other, operator.sub)

    def __rsub__(self, other: Any) -> "ndarray":
        arr_other = asarray(other)
        return arr_other._binary_op(self, operator.sub)

    def __mul__(self, other: Any) -> "ndarray":
        return self._binary_op(other, operator.mul)

    def __rmul__(self, other: Any) -> "ndarray":
        return self.__mul__(other)

    def __truediv__(self, other: Any) -> "ndarray":
        return self._binary_op(other, operator.truediv)

    def __rtruediv__(self, other: Any) -> "ndarray":
        arr_other = asarray(other)
        return arr_other._binary_op(self, operator.truediv)

    def __neg__(self) -> "ndarray":
        return self._unary_op(operator.neg)

    # Comparisons return boolean arrays
    def __lt__(self, other: Any) -> "ndarray":
        return self._binary_op(other, operator.lt)

    def __le__(self, other: Any) -> "ndarray":
        return self._binary_op(other, operator.le)

    def __gt__(self, other: Any) -> "ndarray":
        return self._binary_op(other, operator.gt)

    def __ge__(self, other: Any) -> "ndarray":
        return self._binary_op(other, operator.ge)

    def __eq__(self, other: Any) -> "ndarray":  # type: ignore[override]
        return self._binary_op(other, operator.eq)

    def __ne__(self, other: Any) -> "ndarray":  # type: ignore[override]
        return self._binary_op(other, operator.ne)

    def __matmul__(self, other: Any) -> "ndarray":
        right = asarray(other)
        if self.ndim == 1 and right.ndim == 1:
            if self.shape[0] != right.shape[0]:
                raise ValueError("shapes not aligned for dot product")
            total = math.fsum(float(a) * float(b) for a, b in zip(self.tolist(), right.tolist()))
            return ndarray(total)
        if self.ndim != 2 or right.ndim != 2:
            raise ValueError("matrix multiplication requires two 2D arrays")
        rows, shared = self.shape
        shared_b, cols = right.shape
        if shared != shared_b:
            raise ValueError("shapes not aligned for matmul")
        a = self._data
        b = right._data
        result = []
        for r in range(rows):
            row = []
            for c in range(cols):
                row.append(math.fsum(float(a[r][k]) * float(b[k][c]) for k in range(shared)))
            result.append(row)
        return ndarray(result)


def _expand_axis(data: Any, axis: int) -> Any:
    if axis == 0:
        return [_deep_copy(data)]
    if not isinstance(data, list):
        raise ValueError("cannot expand axis beyond array dimensionality")
    return [_expand_axis(item, axis - 1) for item in data]


def asarray(obj: Any, dtype: Optional[type] = None) -> ndarray:
    if isinstance(obj, ndarray) and dtype is None:
        return ndarray(obj._data)
    nested = _to_nested_list(obj)

    def _cast(value: Any) -> Any:
        if isinstance(value, list):
            return [_cast(v) for v in value]
        if dtype is None:
            if isinstance(value, bool):
                return bool(value)
            if isinstance(value, float):
                return float(value)
            if isinstance(value, int):
                return int(value)
            return value
        if dtype is float:
            return float(value)
        if dtype is bool:
            return bool(value)
        if dtype is int:
            return int(value)
        raise TypeError("unsupported dtype")

    casted = _cast(nested)
    return ndarray(casted)


def array(obj: Any, dtype: Optional[type] = None) -> ndarray:
    return asarray(obj, dtype=dtype)


def zeros(shape: Union[int, Tuple[int, ...]], dtype: type = float) -> ndarray:
    if isinstance(shape, int):
        shape = (shape,)

    def _build(current_shape: Tuple[int, ...]) -> Any:
        if not current_shape:
            return dtype()
        return [_build(current_shape[1:]) for _ in range(current_shape[0])]

    return ndarray(_build(tuple(shape)))


def zeros_like(a: ndarray, dtype: Optional[type] = None) -> ndarray:
    dtype = dtype or float
    return zeros(a.shape, dtype=dtype)


def empty_like(a: ndarray, dtype: Optional[type] = None) -> ndarray:
    return zeros_like(a, dtype=dtype)


def full(shape: Union[int, Tuple[int, ...]], fill_value: Any, dtype: Optional[type] = None) -> ndarray:
    if isinstance(shape, int):
        shape = (shape,)
    dtype = dtype or type(fill_value)

    def _build(current_shape: Tuple[int, ...]) -> Any:
        if not current_shape:
            return dtype(fill_value) if dtype in (float, int) else fill_value
        return [_build(current_shape[1:]) for _ in range(current_shape[0])]

    return ndarray(_build(tuple(shape)))


def full_like(a: ndarray, fill_value: Any, dtype: Optional[type] = None) -> ndarray:
    return full(a.shape, fill_value, dtype=dtype)


def concatenate(arrays: Sequence[Any], axis: int = 0) -> ndarray:
    if not arrays:
        raise ValueError("need at least one array to concatenate")
    converted = [asarray(a) for a in arrays]
    first_shape = converted[0].shape
    if axis < 0:
        axis += len(first_shape)
    if not converted[0].shape:
        # 0-d scalars: treat as 1-d
        axis = 0
    if axis != 0:
        raise NotImplementedError("only axis=0 is supported in this minimal implementation")
    if len(first_shape) == 1:
        data: List[Any] = []
        for arr in converted:
            if arr.ndim != 1:
                raise ValueError("cannot concatenate arrays with different dimensions")
            data.extend(arr.tolist())
        return ndarray(data)
    if len(first_shape) == 2:
        result: List[Any] = []
        for arr in converted:
            if arr.ndim != 2 or arr.shape[1] != first_shape[1]:
                raise ValueError("cannot concatenate arrays with incompatible shapes")
            result.extend(arr.tolist())
        return ndarray(result)
    raise NotImplementedError("concatenate only implemented for 1D and 2D arrays")


def _flatten(data: Any) -> List[float]:
    if isinstance(data, ndarray):
        return _flatten(data._data)
    if isinstance(data, list):
        flat: List[float] = []
        for item in data:
            flat.extend(_flatten(item))
        return flat
    if hasattr(data, "__iter__") and not isinstance(data, (str, bytes)):
        flat: List[float] = []
        for item in data:
            flat.extend(_flatten(item))
        return flat
    return [float(data)]


def sum(a: Any, axis: Optional[int] = None) -> float:
    arr = asarray(a)
    if axis is None:
        return float(math.fsum(_flatten(arr)))
    axis = _normalize_axis(axis, arr.ndim)
    if arr.ndim == 1:
        return float(math.fsum(arr.tolist()))
    if arr.ndim == 2:
        if axis == 0:
            cols = arr.shape[1]
            totals = [0.0] * cols
            for row in arr.tolist():
                for c in range(cols):
                    totals[c] += float(row[c])
            return ndarray(totals)
        if axis == 1:
            return ndarray([float(math.fsum(row)) for row in arr.tolist()])
    raise NotImplementedError("sum currently supports axis=None,0,1 for up to 2D arrays")


def mean(a: Any, axis: Optional[int] = None) -> Any:
    arr = asarray(a)
    if axis is None:
        return float(sum(arr) / arr.size)
    axis = _normalize_axis(axis, arr.ndim)
    if arr.ndim == 1:
        return float(sum(arr) / arr.shape[0])
    if arr.ndim == 2:
        if axis == 0:
            cols = arr.shape[1]
            totals = [0.0] * cols
            for row in arr.tolist():
                for c in range(cols):
                    totals[c] += float(row[c])
            return ndarray([t / arr.shape[0] for t in totals])
        if axis == 1:
            return ndarray([float(math.fsum(row) / len(row)) for row in arr.tolist()])
    raise NotImplementedError("mean currently supports axis=None,0,1 for up to 2D arrays")


def var(a: Any, axis: Optional[int] = None) -> Any:
    arr = asarray(a)
    if axis is None:
        m = mean(arr)
        return float(math.fsum((float(x) - m) ** 2 for x in _flatten(arr)) / arr.size)
    axis = _normalize_axis(axis, arr.ndim)
    if arr.ndim == 1:
        m = mean(arr)
        return float(math.fsum((float(x) - m) ** 2 for x in arr.tolist()) / arr.shape[0])
    if arr.ndim == 2:
        if axis == 0:
            m = mean(arr, axis=0).tolist()
            cols = arr.shape[1]
            totals = [0.0] * cols
            for row in arr.tolist():
                for c in range(cols):
                    diff = float(row[c]) - m[c]
                    totals[c] += diff * diff
            return ndarray([t / arr.shape[0] for t in totals])
        if axis == 1:
            result = []
            for row in arr.tolist():
                m = math.fsum(row) / len(row)
                result.append(math.fsum((float(x) - m) ** 2 for x in row) / len(row))
            return ndarray(result)
    raise NotImplementedError("var currently supports axis=None,0,1 for up to 2D arrays")


def clip(a: Any, min_value: float, max_value: float) -> ndarray:
    arr = asarray(a)

    def _rec(x: Any) -> Any:
        if isinstance(x, list):
            return [_rec(v) for v in x]
        return min(max_value, max(min_value, float(x)))

    return ndarray(_rec(arr._data))


def dot(a: Any, b: Any) -> float:
    arr_a = asarray(a)
    arr_b = asarray(b)
    if arr_a.ndim != 1 or arr_b.ndim != 1:
        raise ValueError("dot currently implemented for 1D arrays only")
    if arr_a.shape[0] != arr_b.shape[0]:
        raise ValueError("shapes not aligned for dot")
    return float(math.fsum(float(x) * float(y) for x, y in zip(arr_a.tolist(), arr_b.tolist())))


def cumsum(a: Any) -> ndarray:
    arr = asarray(a)
    if arr.ndim != 1:
        raise NotImplementedError("cumsum currently supports 1D arrays only")
    total = 0.0
    result: List[float] = []
    for x in arr.tolist():
        total += float(x)
        result.append(total)
    return ndarray(result)


def arange(start: Number, stop: Optional[Number] = None, step: Number = 1, dtype: Optional[type] = None) -> ndarray:
    if stop is None:
        start, stop = 0, start
    values: List[float] = []
    current = float(start)
    assert stop is not None
    if step == 0:
        raise ValueError("step must be non-zero")
    if step > 0:
        while current < float(stop):
            values.append(current)
            current += float(step)
    else:
        while current > float(stop):
            values.append(current)
            current += float(step)
    if dtype is float:
        values = [float(v) for v in values]
    elif dtype is int:
        values = [int(v) for v in values]
    return ndarray(values)


def argsort(a: Any, kind: str = "quicksort") -> ndarray:
    arr = asarray(a)
    if arr.ndim != 1:
        raise ValueError("argsort currently supports 1D arrays only")
    indexed = list(enumerate(arr.tolist()))
    indexed.sort(key=lambda kv: kv[1])
    return ndarray([idx for idx, _ in indexed])


def unique(a: Any, return_counts: bool = False) -> Union[ndarray, Tuple[ndarray, ndarray]]:
    arr = asarray(a)
    if arr.ndim != 1:
        raise ValueError("unique currently supports 1D arrays only")
    counts = Counter(arr.tolist())
    values = sorted(counts.keys())
    if return_counts:
        return ndarray(values), ndarray([counts[v] for v in values])
    return ndarray(values)


def median(a: Any) -> float:
    arr = asarray(a)
    flat = sorted(_flatten(arr))
    n = len(flat)
    if n == 0:
        raise ValueError("median of empty array")
    mid = n // 2
    if n % 2:
        return float(flat[mid])
    return float((flat[mid - 1] + flat[mid]) / 2.0)


def all(a: Any) -> bool:
    arr = asarray(a)
    return builtins.all(bool(x) for x in _flatten(arr))


def isfinite(a: Any) -> Union[bool, ndarray]:
    if isinstance(a, (int, float)):
        return math.isfinite(a)
    arr = asarray(a)

    def _rec(x: Any) -> Any:
        if isinstance(x, list):
            return [_rec(v) for v in x]
        return math.isfinite(float(x))

    return ndarray(_rec(arr._data))


def linspace(start: float, stop: float, num: int, dtype: Optional[type] = None) -> ndarray:
    if num <= 0:
        raise ValueError("num must be positive")
    if num == 1:
        values = [start]
    else:
        step = (stop - start) / (num - 1)
        values = [start + step * i for i in range(num)]
    if dtype is float:
        values = [float(v) for v in values]
    return ndarray(values)


def expand_dims(a: Any, axis: int) -> ndarray:
    arr = asarray(a)
    if axis < 0:
        axis += arr.ndim + 1

    def _rec(data: Any, depth: int) -> Any:
        if depth == 0:
            return [_deep_copy(data)]
        if not isinstance(data, list):
            raise ValueError("cannot expand dims beyond data depth")
        return [_rec(item, depth - 1) for item in data]

    return ndarray(_rec(arr._data, axis))


def _norm(values: Iterable[float]) -> float:
    return math.sqrt(math.fsum(float(v) ** 2 for v in values))


class _LinalgModule:
    def norm(self, a: Any, axis: Optional[int] = None, keepdims: bool = False) -> Any:
        arr = asarray(a)
        if axis is None:
            return _norm(_flatten(arr))
        axis = _normalize_axis(axis, arr.ndim)
        if arr.ndim == 1:
            n = _norm(arr.tolist())
            if keepdims:
                return ndarray([[n]]) if axis == 0 else ndarray([n])
            return n
        if arr.ndim == 2:
            if axis == 0:
                cols = arr.shape[1]
                result = []
                for c in range(cols):
                    col_vals = [float(row[c]) for row in arr.tolist()]
                    result.append(_norm(col_vals))
                if keepdims:
                    return ndarray([result])
                return ndarray(result)
            if axis == 1:
                result = []
                for row in arr.tolist():
                    result.append(_norm(row))
                if keepdims:
                    return ndarray([[v] for v in result])
                return ndarray(result)
        raise NotImplementedError("norm currently supports up to 2D arrays")


class _Generator:
    def __init__(self, seed: Optional[int] = None):
        self._rng = _py_random.Random(seed)

    def normal(
        self,
        loc: float = 0.0,
        scale: float = 1.0,
        size: Optional[Union[int, Tuple[int, ...]]] = None,
    ) -> Any:
        def _sample() -> float:
            return self._rng.gauss(loc, scale)

        if size is None:
            return _sample()
        if isinstance(size, int):
            return ndarray([_sample() for _ in range(size)])
        if isinstance(size, tuple):
            if not size:
                return ndarray(_sample())

            def _build(dim: Tuple[int, ...]) -> Any:
                if not dim:
                    return _sample()
                return [_build(dim[1:]) for _ in range(dim[0])]

            return ndarray(_build(size))
        raise TypeError("size must be int, tuple, or None")

    def permutation(self, x: Union[int, Sequence[Any]]) -> ndarray:
        if isinstance(x, int):
            arr = list(range(x))
        else:
            arr = list(x)
        self._rng.shuffle(arr)
        return ndarray(arr)


class _RandomModule:
    def default_rng(self, seed: Optional[int] = None) -> _Generator:
        return _Generator(seed)


linalg = _LinalgModule()
random = _RandomModule()

inf = float("inf")
newaxis = None
ndarray = ndarray

__all__ = [
    "array",
    "asarray",
    "zeros",
    "zeros_like",
    "empty_like",
    "full",
    "full_like",
    "concatenate",
    "sum",
    "mean",
    "var",
    "clip",
    "dot",
    "cumsum",
    "arange",
    "argsort",
    "unique",
    "median",
    "all",
    "isfinite",
    "linspace",
    "expand_dims",
    "linalg",
    "random",
    "inf",
    "newaxis",
    "ndarray",
]
