from math import floor
from typing import Callable, NamedTuple, Union

import numpy as np

Mat = np.ndarray

_IntOrFloat = Union[int, float]


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

    def __add__(self, other):
        if not isinstance(other, (list, tuple)) or len(other) != 4:
            raise TypeError

        return self.__class__(*[a + b for a, b in zip(self, other)])

    def __sub__(self, other):
        if not isinstance(other, (list, tuple)) or len(other) != 4:
            raise TypeError

        return self.__class__(*[a - b for a, b in zip(self, other)])

    def __mul__(self, other):
        if not isinstance(other, (int, float)):
            raise TypeError

        return self.__class__(*[v * other for v in self])
