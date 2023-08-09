from math import floor
from typing import Any, Tuple

import numpy as np

from ...types import Mat
from .definition import Device

__all__ = [
    "crop_img",
    "crop_from_device_attr",
    "crop_to_pure",
    "crop_to_far",
    "crop_to_lost",
    "crop_to_max_recall",
    "crop_to_rating_class",
    "crop_to_score",
    "crop_to_title",
    "crop_black_edges",
]


def crop_img(img: Mat, *, top: int, left: int, bottom: int, right: int):
    return img[top:bottom, left:right]


def crop_from_device_attr(img: Mat, rect: Tuple[int, int, int, int]):
    x, y, w, h = rect
    return crop_img(img, top=y, left=x, bottom=y + h, right=x + w)


def crop_to_pure(screenshot: Mat, device: Device):
    return crop_from_device_attr(screenshot, device.pure)


def crop_to_far(screenshot: Mat, device: Device):
    return crop_from_device_attr(screenshot, device.far)


def crop_to_lost(screenshot: Mat, device: Device):
    return crop_from_device_attr(screenshot, device.lost)


def crop_to_max_recall(screenshot: Mat, device: Device):
    return crop_from_device_attr(screenshot, device.max_recall)


def crop_to_rating_class(screenshot: Mat, device: Device):
    return crop_from_device_attr(screenshot, device.rating_class)


def crop_to_score(screenshot: Mat, device: Device):
    return crop_from_device_attr(screenshot, device.score)


def crop_to_title(screenshot: Mat, device: Device):
    return crop_from_device_attr(screenshot, device.title)


def is_black_edge(list_of_pixels: Mat, black_pixel=None):
    if black_pixel is None:
        black_pixel = np.array([0, 0, 0], list_of_pixels.dtype)
    pixels = list_of_pixels.reshape([-1, 3])
    return np.count_nonzero(all(pixels < black_pixel, axis=1)) > floor(
        len(pixels) * 0.6
    )
