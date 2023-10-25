from collections.abc import Iterable
from typing import NamedTuple, Tuple, Union

import numpy as np

Mat = np.ndarray


class XYWHRect(NamedTuple):
    x: int
    y: int
    w: int
    h: int

    def __add__(self, other: Union["XYWHRect", Tuple[int, int, int, int]]):
        if not isinstance(other, Iterable) or len(other) != 4:
            raise ValueError()

        return self.__class__(*[a + b for a, b in zip(self, other)])

    def __sub__(self, other: Union["XYWHRect", Tuple[int, int, int, int]]):
        if not isinstance(other, Iterable) or len(other) != 4:
            raise ValueError()

        return self.__class__(*[a - b for a, b in zip(self, other)])
