import math
from typing import Tuple

import cv2
import numpy as np

from .types import Mat

__all__ = ["crop_xywh", "CropBlackEdges"]


def crop_xywh(mat: Mat, rect: Tuple[int, int, int, int]):
    x, y, w, h = rect
    return mat[y : y + h, x : x + w]


class CropBlackEdges:
    @staticmethod
    def is_black_edge(__img_gray_slice: Mat, black_pixel: int, ratio: float = 0.6):
        pixels_compared = __img_gray_slice < black_pixel
        return np.count_nonzero(pixels_compared) > math.floor(
            __img_gray_slice.size * ratio
        )

    @classmethod
    def get_crop_rect(cls, img_gray: Mat, black_threshold: int = 25):
        height, width = img_gray.shape[:2]
        left = 0
        right = width
        top = 0
        bottom = height

        for i in range(width):
            column = img_gray[:, i]
            if not cls.is_black_edge(column, black_threshold):
                break
            left += 1

        for i in sorted(range(width), reverse=True):
            column = img_gray[:, i]
            if i <= left + 1 or not cls.is_black_edge(column, black_threshold):
                break
            right -= 1

        for i in range(height):
            row = img_gray[i]
            if not cls.is_black_edge(row, black_threshold):
                break
            top += 1

        for i in sorted(range(height), reverse=True):
            row = img_gray[i]
            if i <= top + 1 or not cls.is_black_edge(row, black_threshold):
                break
            bottom -= 1

        assert right > left, "cropped width < 0"
        assert bottom > top, "cropped height < 0"
        return (left, top, right - left, bottom - top)

    @classmethod
    def crop(
        cls, img: Mat, convert_flag: cv2.COLOR_BGR2GRAY, black_threshold: int = 25
    ) -> Mat:
        rect = cls.get_crop_rect(cv2.cvtColor(img, convert_flag), black_threshold)
        return crop_xywh(img, rect)
