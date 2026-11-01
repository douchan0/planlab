"""Minimal, dependency-free reader for the JSON-stat 2.0 format used by Eurostat.

Eurostat's dissemination API returns JSON-stat 2.0 datasets whose ``value`` member
may be *dense* (a JSON list, possibly containing ``null``) or *sparse* (a JSON
object keyed by the stringified flat index of each defined cell).  Both are
supported here.

The flat index is **dimension-major in the order given by the ``id`` member**:
the right-most dimension varies fastest.  For a dataset with ``id = [a, b, c]``
and ``size = [2, 3, 4]`` the cell ``(i, j, k)`` lives at flat index
``i*12 + j*4 + k``.  This module never assumes a particular dimension order; it
reads ``id``/``size`` from the payload, which is what makes it robust to
Eurostat re-ordering or filtering dimensions.

Public API
----------
``JsonStat.parse(payload)``
    Parse a JSON-stat payload (dict, bytes or str) into a :class:`JsonStat`.
``JsonStat.code(dim, position)`` / ``JsonStat.position(dim, code)``
    Translate between category codes and positions along a dimension.
``JsonStat.dense(dim_order=None)``
    Materialise the data as an ``numpy.ndarray`` of shape ``size``; undefined
    cells become ``nan``.
``JsonStat.single(dim)``
    Return the only category code present on a size-1 dimension (or ``None``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

__all__ = ["JsonStat", "JsonStatError"]


class JsonStatError(ValueError):
    """Raised when a payload is not valid JSON-stat 2.0 for our purposes."""


@dataclass(frozen=True)
class JsonStat:
    """An immutable, parsed JSON-stat 2.0 dataset.

    Attributes
    ----------
    label:
        Human readable dataset label from the payload.
    updated:
        ``updated`` timestamp string from the payload (may be ``None``).
    ids:
        Dimension ids, in ``id`` order (the order that defines the flat index).
    sizes:
        Size of each dimension, aligned with :attr:`ids`.
    codes:
        ``{dim: tuple[code, ...]}`` category codes in payload order.
    labels:
        ``{dim: {code: label}}``.
    values:
        ``{flat_index: float}`` for every defined cell.  Undefined cells are
        simply absent (JSON-stat sparse encoding).
    statuses:
        ``{flat_index: str}`` observation status flags (``e`` estimated, ``p``
        provisional, ``c`` confidential, ...), when supplied.
    extension:
        The ``extension`` member verbatim (carries the datastructure version).
    source_label:
        ``source`` member (usually ``"ESTAT"``).
    """

    label: str
    updated: str | None
    ids: tuple[str, ...]
    sizes: tuple[int, ...]
    codes: Mapping[str, tuple[str, ...]]
    labels: Mapping[str, Mapping[str, str]]
    values: Mapping[int, float]
    statuses: Mapping[int, str] = field(default_factory=dict)
    extension: Mapping[str, Any] = field(default_factory=dict)
    source_label: str | None = None

    # ------------------------------------------------------------------ parse
    @classmethod
    def parse(cls, payload: str | bytes | Mapping[str, Any]) -> "JsonStat":
        """Parse a JSON-stat 2.0 payload.

        Args:
            payload: Raw response body (``bytes``/``str``) or an already
                decoded ``dict``.

        Returns:
            A :class:`JsonStat` instance.

        Raises:
            JsonStatError: if required members are missing or inconsistent.
        """
        if isinstance(payload, (str, bytes, bytearray)):
            obj = _loads(payload)
        else:
            obj = dict(payload)
        if not isinstance(obj, Mapping):
            raise JsonStatError("JSON-stat payload is not a JSON object")

        ids = obj.get("id")
        sizes = obj.get("size")
        if not ids or not sizes:
            raise JsonStatError("JSON-stat payload lacks 'id'/'size' members")
        ids = tuple(str(d) for d in ids)
        sizes = tuple(int(s) for s in sizes)
        if len(ids) != len(sizes):
            raise JsonStatError(f"len(id)={len(ids)} != len(size)={len(sizes)}")

        dims = obj.get("dimension")
        if not isinstance(dims, Mapping):
            raise JsonStatError("JSON-stat payload lacks a 'dimension' object")

        codes: dict[str, tuple[str, ...]] = {}
        labels: dict[str, dict[str, str]] = {}
        for dim in ids:
            spec = dims.get(dim)
            if spec is None:
                raise JsonStatError(f"dimension {dim!r} listed in 'id' is absent")
            category = spec.get("category", {})
            codes[dim] = _category_codes(category, dim)
            labels[dim] = dict(category.get("label", {}) or {})
            if len(codes[dim]) != sizes[ids.index(dim)]:
                raise JsonStatError(
                    f"dimension {dim!r} has {len(codes[dim])} categories "
                    f"but size={sizes[ids.index(dim)]}"
                )

        strides = _strides(sizes)
        values = _values(obj.get("value"), strides, ids, sizes)
        statuses = _statuses(obj.get("status"))

        return cls(
            label=str(obj.get("label", "")),
            updated=obj.get("updated"),
            ids=ids,
            sizes=sizes,
            codes=codes,
            labels=labels,
            values=values,
            statuses=statuses,
            extension=dict(obj.get("extension", {}) or {}),
            source_label=obj.get("source"),
        )

    # ------------------------------------------------------------- accessors
    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return len(self.ids)

    @property
    def n_cells(self) -> int:
        """Total number of cells implied by ``size``."""
        return int(np.prod(self.sizes)) if self.sizes else 0

    def position(self, dim: str, code: str) -> int:
        """Return the zero-based position of ``code`` along ``dim``.

        Raises:
            KeyError: if the dimension or the code is unknown.
        """
        try:
            return self.codes[dim].index(code)
        except (KeyError, ValueError) as exc:  # pragma: no cover - defensive
            raise KeyError(f"{code!r} is not a category of dimension {dim!r}") from exc

    def code(self, dim: str, position: int) -> str:
        """Return the category code at ``position`` along ``dim``."""
        return self.codes[dim][position]

    def single(self, dim: str) -> str | None:
        """Return the sole code of a size-1 dimension, else ``None``."""
        seq = self.codes.get(dim, ())
        return seq[0] if len(seq) == 1 else None

    def dense(self, fill: float = np.nan) -> np.ndarray:
        """Materialise the data as an array of shape :attr:`sizes`.

        Args:
            fill: Value used for undefined (absent) cells.

        Returns:
            ``numpy.ndarray`` with ``size`` shape, C order (``id`` order).
        """
        out = np.full(self.sizes, fill, dtype=float)
        if not self.values:
            return out
        flat_idx = np.fromiter(self.values.keys(), dtype=np.int64, count=len(self.values))
        vals = np.fromiter(self.values.values(), dtype=float, count=len(self.values))
        out.reshape(-1)[flat_idx] = vals
        return out

    def defined(self) -> int:
        """Number of defined (non-null) cells."""
        return len(self.values)

    def status_counts(self) -> dict[str, int]:
        """Histogram of observation-status flags."""
        counts: dict[str, int] = {}
        for flag in self.statuses.values():
            key = flag or "missing"
            counts[key] = counts.get(key, 0) + 1
        return counts


# --------------------------------------------------------------------- helpers
def _loads(payload: str | bytes | bytearray) -> Any:
    if isinstance(payload, (bytes, bytearray)):
        payload = payload.decode("utf-8")
    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise JsonStatError(f"response is not valid JSON: {exc}") from exc


def _category_codes(category: Mapping[str, Any], dim: str) -> tuple[str, ...]:
    """Extract category codes in payload order from a ``category`` object."""
    index = category.get("index")
    if index is None:
        raise JsonStatError(f"dimension {dim!r} has no 'category.index'")
    if isinstance(index, Mapping):
        # ``index`` maps code -> position; keep the payload order by position.
        ordered = sorted(index.items(), key=lambda kv: int(kv[1]))
        return tuple(str(k) for k, _ in ordered)
    if isinstance(index, Sequence):
        return tuple(str(k) for k in index)
    raise JsonStatError(f"dimension {dim!r} has an unsupported 'category.index'")


def _strides(sizes: Sequence[int]) -> tuple[int, ...]:
    strides = [1] * len(sizes)
    for i in range(len(sizes) - 2, -1, -1):
        strides[i] = strides[i + 1] * sizes[i + 1]
    return tuple(strides)


def _values(
    raw: Any,
    strides: Sequence[int],
    ids: Sequence[str],
    sizes: Sequence[int],
) -> dict[int, float]:
    """Flatten JSON-stat ``value`` into ``{flat_index: float}``."""
    n = int(np.prod(sizes)) if sizes else 0
    if raw is None:
        return {}
    out: dict[int, float] = {}
    if isinstance(raw, Mapping):
        for key, val in raw.items():
            if val is None:
                continue
            idx = int(key)
            if idx < 0 or idx >= n:
                raise JsonStatError(f"flat index {idx} outside [0, {n})")
            out[idx] = float(val)
        return out
    if isinstance(raw, Sequence):
        for idx, val in enumerate(raw):
            if val is None:
                continue
            out[idx] = float(val)
        return out
    raise JsonStatError(f"unsupported 'value' member of type {type(raw).__name__}")


def _statuses(raw: Any) -> dict[int, str]:
    if not isinstance(raw, Mapping):
        return {}
    return {int(k): str(v) for k, v in raw.items()}
