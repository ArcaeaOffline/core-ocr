from collections.abc import Iterable
from typing import Callable, Tuple, TypeVar, Union, overload

from cv2 import IMREAD_UNCHANGED, imdecode
from numpy import fromfile as np_fromfile
from numpy import uint8

from .types import Mat, XYWHRect

__all__ = ["imread_unicode"]


def imread_unicode(filepath: str) -> Mat:
    # https://stackoverflow.com/a/57872297/16484891
    # CC BY-SA 4.0
    return imdecode(np_fromfile(filepath, dtype=uint8), IMREAD_UNCHANGED)


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
    elif isinstance(item, Iterable):
        return item.__class__([i * factor for i in item])
