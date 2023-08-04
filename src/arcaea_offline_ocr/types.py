from collections.abc import Iterable
from typing import Any, NamedTuple, Protocol, Tuple, Union

import numpy as np

# from pylance
Mat = np.ndarray[int, np.dtype[np.generic]]


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


class cv2_ml_StatModel(Protocol):
    def predict(self, samples: np.ndarray, results: np.ndarray, flags: int = 0):
        ...

    def train(self, samples: np.ndarray, layout: int, responses: np.ndarray):
        ...


class cv2_ml_KNearest(cv2_ml_StatModel, Protocol):
    def findNearest(
        self, samples: np.ndarray, k: int
    ) -> Tuple[Any, np.ndarray, np.ndarray, np.ndarray]:
        """cv.ml.KNearest.findNearest(samples, k[, results[, neighborResponses[, dist]]]) -> retval, results, neighborResponses, dist"""
        ...
