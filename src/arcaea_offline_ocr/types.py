from collections.abc import Callable, Iterable
from math import floor
from typing import Any, NamedTuple

_IntOrFloat = int | float


class XYWHRect(NamedTuple):
    x: _IntOrFloat
    y: _IntOrFloat
    w: _IntOrFloat
    h: _IntOrFloat

    def _to_int(self, func: Callable[[_IntOrFloat], int]):
        return (func(self.x), func(self.y), func(self.w), func(self.h))

    def rounded(self):
        return self._to_int(round)

    def floored(self):
        return self._to_int(floor)

    # tuple's operators accept arbitrary iterables; the rect domain only allows
    # 4-element numeric sequences, so the isinstance guard rejects other shapes.
    def __add__(self, other: Iterable[Any]):
        if not isinstance(other, (list, tuple)) or len(other) != 4:
            raise TypeError

        return self.__class__(*[a + b for a, b in zip(self, other, strict=False)])

    def __sub__(self, other: Iterable[Any]):
        if not isinstance(other, (list, tuple)) or len(other) != 4:
            raise TypeError

        return self.__class__(*[a - b for a, b in zip(self, other, strict=False)])

    def __mul__(self, other: Any):
        if not isinstance(other, (int, float)):
            raise TypeError

        return self.__class__(*[v * other for v in self])
