from collections.abc import Iterable
from typing import Callable, TypeVar, Union, overload

import cv2
import numpy as np

from .types import XYWHRect

__all__ = ["imread_unicode"]


def imread_unicode(filepath: str, flags: int = cv2.IMREAD_UNCHANGED):
    # https://stackoverflow.com/a/57872297/16484891
    # CC BY-SA 4.0
    return cv2.imdecode(np.fromfile(filepath, dtype=np.uint8), flags)


def construct_int_xywh_rect(
    rect: XYWHRect, func: Callable[[Union[int, float]], int] = round
):
    return XYWHRect(*[func(num) for num in rect])


@overload
def apply_factor(item: int, factor: float) -> float:
    ...


@overload
def apply_factor(item: float, factor: float) -> float:
    ...


T = TypeVar("T", bound=Iterable)


@overload
def apply_factor(item: T, factor: float) -> T:
    ...


def apply_factor(item, factor: float):
    if isinstance(item, (int, float)):
        return item * factor
    if isinstance(item, Iterable):
        return item.__class__([i * factor for i in item])
