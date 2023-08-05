from math import floor
from typing import Tuple

from numpy import all, array, count_nonzero

from .types import Mat

__all__ = ["crop_xywh", "crop_black_edges"]


def crop_xywh(mat: Mat, rect: Tuple[int, int, int, int]):
    x, y, w, h = rect
    return mat[y : y + h, x : x + w]


def is_black_edge(list_of_pixels: Mat, black_pixel=None):
    if black_pixel is None:
        black_pixel = array([0, 0, 0], list_of_pixels.dtype)
    pixels = list_of_pixels.reshape([-1, 3])
    return count_nonzero(all(pixels < black_pixel, axis=1)) > floor(len(pixels) * 0.6)


def crop_black_edges(screenshot: Mat):
    cropped = screenshot.copy()
    black_pixel = array([50, 50, 50], screenshot.dtype)
    height, width = screenshot.shape[:2]
    left = 0
    right = width
    top = 0
    bottom = height

    for i in range(width):
        column = cropped[:, i]
        if not is_black_edge(column, black_pixel):
            break
        left += 1

    for i in sorted(range(width), reverse=True):
        column = cropped[:, i]
        if i <= left + 1 or not is_black_edge(column, black_pixel):
            break
        right -= 1

    for i in range(height):
        row = cropped[i]
        if not is_black_edge(row, black_pixel):
            break
        top += 1

    for i in sorted(range(height), reverse=True):
        row = cropped[i]
        if i <= top + 1 or not is_black_edge(row, black_pixel):
            break
        bottom -= 1

    return cropped[top:bottom, left:right]
