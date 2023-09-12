from math import floor
from typing import Tuple

import numpy as np

from .types import Mat

__all__ = ["crop_xywh", "crop_black_edges", "crop_black_edges_grayscale"]


def crop_xywh(mat: Mat, rect: Tuple[int, int, int, int]):
    x, y, w, h = rect
    return mat[y : y + h, x : x + w]


def is_black_edge(list_of_pixels: Mat, black_pixel: Mat, ratio: float = 0.6):
    pixels = list_of_pixels.reshape([-1, 3])
    return np.count_nonzero(np.all(pixels < black_pixel, axis=1)) > floor(
        len(pixels) * ratio
    )


def crop_black_edges(img_bgr: Mat, black_threshold: int = 50):
    cropped = img_bgr.copy()
    black_pixel = np.array([black_threshold] * 3, img_bgr.dtype)
    height, width = img_bgr.shape[:2]
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


def is_black_edge_grayscale(
    gray_value_list: np.ndarray, black_threshold: int = 50, ratio: float = 0.6
) -> bool:
    return (
        np.count_nonzero(gray_value_list < black_threshold)
        > len(gray_value_list) * ratio
    )


def crop_black_edges_grayscale(
    img_gray: Mat, black_threshold: int = 50
) -> Tuple[int, int, int, int]:
    """Returns cropped rect"""
    height, width = img_gray.shape[:2]
    left = 0
    right = width
    top = 0
    bottom = height

    for i in range(width):
        column = img_gray[:, i]
        if not is_black_edge_grayscale(column, black_threshold):
            break
        left += 1

    for i in sorted(range(width), reverse=True):
        column = img_gray[:, i]
        if i <= left + 1 or not is_black_edge_grayscale(column, black_threshold):
            break
        right -= 1

    for i in range(height):
        row = img_gray[i]
        if not is_black_edge_grayscale(row, black_threshold):
            break
        top += 1

    for i in sorted(range(height), reverse=True):
        row = img_gray[i]
        if i <= top + 1 or not is_black_edge_grayscale(row, black_threshold):
            break
        bottom -= 1

    assert right > left, "cropped width > 0"
    assert bottom > top, "cropped height > 0"
    return (left, top, right - left, bottom - top)
