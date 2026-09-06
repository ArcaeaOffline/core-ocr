from typing import Any

import cv2
import numpy as np

from ._common import resize_image


def average(
    img_gray: cv2.typing.MatLike,
    hash_size: int,
) -> np.ndarray[tuple[Any, ...], np.dtype[np.bool]]:
    img_resized = resize_image(img_gray, (hash_size, hash_size))
    diff = img_resized > img_resized.mean()
    return diff.flatten()


def difference(
    img_gray: cv2.typing.MatLike,
    hash_size: int,
) -> np.ndarray[tuple[Any, ...], np.dtype[np.bool]]:
    img_size = (hash_size + 1, hash_size)
    img_resized = resize_image(img_gray, img_size)

    previous = img_resized[:, :-1]
    current = img_resized[:, 1:]
    diff = previous > current
    return diff.flatten()


def dct(
    img_gray: cv2.typing.MatLike,
    hash_size: int = 16,
    high_freq_factor: int = 4,
) -> np.ndarray[tuple[Any, ...], np.dtype[np.bool]]:
    # TODO: consistency?  # noqa: FIX002, TD002, TD003
    img_size_base = hash_size * high_freq_factor
    img_size = (img_size_base, img_size_base)

    img_resized = resize_image(img_gray, img_size)
    img_resized = img_resized.astype(np.float32)
    dct_mat = cv2.dct(img_resized)

    hash_mat = dct_mat[:hash_size, :hash_size]
    # median, following the standard pHash algorithm
    # (hackerfactor "Looks Like It" and imagehash's phash)
    return hash_mat > np.median(hash_mat)
