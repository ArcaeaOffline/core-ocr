import math
from copy import deepcopy
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np
from numpy.linalg import norm

from .crop import crop_xywh
from .mask import mask_byd, mask_ftr, mask_prs, mask_pst
from .types import Mat, cv2_ml_KNearest

__all__ = [
    "FixRects",
    "preprocess_hog",
    "ocr_digits_by_contour_get_samples",
    "ocr_digits_by_contour_knn",
]


class FixRects:
    @staticmethod
    def connect_broken(
        rects: Sequence[Tuple[int, int, int, int]],
        img_width: int,
        img_height: int,
        tolerance: Optional[int] = None,
    ):
        # for a "broken" digit, please refer to
        # /assets/fix_rects/broken_masked.jpg
        # the larger "5" in the image is a "broken" digit

        if tolerance is None:
            tolerance = math.ceil(img_width * 0.08)

        new_rects = []
        consumed_rects = []
        for rect in rects:
            x, y, w, h = rect
            # grab those small rects
            if not img_height * 0.1 <= h <= img_height * 0.6:
                continue

            group = []
            # see if there's other rects that have near left & right borders
            for other_rect in rects:
                if rect == other_rect:
                    continue
                ox, oy, ow, oh = other_rect
                if abs(x - ox) < tolerance and abs((x + w) - (ox + ow)) < tolerance:
                    group.append(other_rect)

            if group:
                group.append(rect)
                consumed_rects.extend(group)
                # calculate the new rect
                new_x = min(r[0] for r in group)
                new_y = min(r[1] for r in group)
                new_right = max(r[0] + r[2] for r in group)
                new_bottom = max(r[1] + r[3] for r in group)
                new_w = new_right - new_x
                new_h = new_bottom - new_y
                new_rects.append((new_x, new_y, new_w, new_h))

        return_rects = deepcopy(rects)
        return_rects = [r for r in return_rects if r not in consumed_rects]
        return_rects.extend(new_rects)
        return return_rects


def resize_fill_sqaure(img, target: int = 20):
    h, w = img.shape[:2]
    if h > w:
        new_h = target
        new_w = round(w * (target / h))
    else:
        new_w = target
        new_h = round(h * (target / w))
    resized = cv2.resize(img, (new_w, new_h))

    border_size = math.ceil((max(new_w, new_h) - min(new_w, new_h)) / 2)
    if new_w < new_h:
        resized = cv2.copyMakeBorder(
            resized, 0, 0, border_size, border_size, cv2.BORDER_CONSTANT
        )
    else:
        resized = cv2.copyMakeBorder(
            resized, border_size, border_size, 0, 0, cv2.BORDER_CONSTANT
        )
    return cv2.resize(resized, (target, target))


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


def ocr_digit_samples_knn(__samples, knn_model: cv2_ml_KNearest, k: int = 4):
    _, results, _, _ = knn_model.findNearest(__samples, k)
    result_list = [int(r) for r in results.ravel()]
    result_str = "".join(str(r) for r in result_list if r > -1)
    return int(result_str)


def ocr_digits_by_contour_get_samples(__roi_gray: Mat, size: int):
    roi = __roi_gray.copy()
    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    rects = sorted([cv2.boundingRect(c) for c in contours], key=lambda r: r[0])
    # digit_rois = [cv2.resize(crop_xywh(roi, rect), size) for rect in rects]
    digit_rois = [resize_fill_sqaure(crop_xywh(roi, rect), size) for rect in rects]
    return preprocess_hog(digit_rois)


def ocr_digits_by_contour_knn(
    __roi_gray: Mat,
    knn_model: cv2_ml_KNearest,
    *,
    k=4,
    size: int = 20,
) -> int:
    samples = ocr_digits_by_contour_get_samples(__roi_gray, size)
    return ocr_digit_samples_knn(samples, knn_model, k)


def ocr_rating_class(roi_hsv: Mat):
    mask_results = [
        mask_pst(roi_hsv),
        mask_prs(roi_hsv),
        mask_ftr(roi_hsv),
        mask_byd(roi_hsv),
    ]
    return max(enumerate(mask_results), key=lambda e: np.count_nonzero(e[1]))[0]
