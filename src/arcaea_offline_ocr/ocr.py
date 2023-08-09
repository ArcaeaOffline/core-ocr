from typing import Tuple

import cv2
import numpy as np
from numpy.linalg import norm

from .crop import crop_xywh
from .mask import mask_byd, mask_ftr, mask_prs, mask_pst
from .types import Mat, cv2_ml_KNearest

__all__ = [
    "preprocess_hog",
    "ocr_digits_by_contour_samples",
    "ocr_digits_by_contour_knn",
]


def preprocess_hog(digit_rois):
    # https://github.com/opencv/opencv/blob/f834736307c8328340aea48908484052170c9224/samples/python/digits.py
    samples = []
    for digit in digit_rois:
        gx = cv2.Sobel(digit, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(digit, cv2.CV_32F, 0, 1)
        mag, ang = cv2.cartToPolar(gx, gy)
        bin_n = 16
        _bin = np.int32(bin_n * ang / (2 * np.pi))
        bin_cells = _bin[:10, :10], _bin[10:, :10], _bin[:10, 10:], _bin[10:, 10:]
        mag_cells = mag[:10, :10], mag[10:, :10], mag[:10, 10:], mag[10:, 10:]
        hists = [
            np.bincount(b.ravel(), m.ravel(), bin_n)
            for b, m in zip(bin_cells, mag_cells)
        ]
        hist = np.hstack(hists)

        # transform to Hellinger kernel
        eps = 1e-7
        hist /= hist.sum() + eps
        hist = np.sqrt(hist)
        hist /= norm(hist) + eps

        samples.append(hist)
    return np.float32(samples)


def ocr_digits_by_contour_samples(__roi_gray: Mat, size: Tuple[int, int]):
    roi = __roi_gray.copy()
    contours = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    rects = sorted([cv2.boundingRect(c) for c in contours], key=lambda r: r[0])
    digit_rois = [cv2.resize(crop_xywh(roi, rect), size) for rect in rects]
    return preprocess_hog(digit_rois)


def ocr_digits_by_contour_knn(
    __roi_gray: Mat,
    knn_model: cv2_ml_KNearest,
    *,
    k=4,
    size: Tuple[int, int] = (20, 20),
) -> int:
    samples = ocr_digits_by_contour_samples(__roi_gray, size)
    _, results, _, _ = knn_model.findNearest(samples, k)
    results = [str(int(i)) for i in results.ravel()]
    return int("".join(results))


def ocr_rating_class(roi_hsv: Mat):
    mask_results = [
        mask_pst(roi_hsv),
        mask_prs(roi_hsv),
        mask_ftr(roi_hsv),
        mask_byd(roi_hsv),
    ]
    return max(enumerate(mask_results), key=lambda e: np.count_nonzero(e[1]))[0]
