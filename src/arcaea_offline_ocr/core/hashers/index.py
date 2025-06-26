import cv2
import numpy as np

from arcaea_offline_ocr.types import Mat

from ._common import _resize_image


def average(img_gray: Mat, hash_size: int) -> Mat:
    img_resized = _resize_image(img_gray, (hash_size, hash_size))
    diff = img_resized > img_resized.mean()
    return diff.flatten()


def difference(img_gray: Mat, hash_size: int) -> Mat:
    img_size = (hash_size + 1, hash_size)
    img_resized = _resize_image(img_gray, img_size)

    previous = img_resized[:, :-1]
    current = img_resized[:, 1:]
    diff = previous > current
    return diff.flatten()


def dct(img_gray: Mat, hash_size: int = 16, high_freq_factor: int = 4) -> Mat:
    # TODO: consistency?  # noqa: FIX002, TD002, TD003
    img_size_base = hash_size * high_freq_factor
    img_size = (img_size_base, img_size_base)

    img_resized = _resize_image(img_gray, img_size)
    img_resized = img_resized.astype(np.float32)
    dct_mat = cv2.dct(img_resized)

    hash_mat = dct_mat[:hash_size, :hash_size]
    return hash_mat > hash_mat.mean()
