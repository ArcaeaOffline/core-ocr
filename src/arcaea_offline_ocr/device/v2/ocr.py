import math
from functools import lru_cache
from typing import Sequence

import cv2
import numpy as np

from ...crop import crop_xywh
from ...mask import mask_byd, mask_ftr, mask_gray, mask_prs, mask_pst, mask_white
from ...ocr import (
    FixRects,
    ocr_digit_samples_knn,
    ocr_digits_by_contour_knn,
    preprocess_hog,
    resize_fill_square,
)
from ...sift_db import SIFTDatabase
from ...types import Mat, cv2_ml_KNearest
from ..shared import DeviceOcrResult
from .preprocess import find_digits_preprocess
from .rois import DeviceV2Rois
from .shared import MAX_RECALL_CLOSE_KERNEL


class DeviceV2Ocr:
    def __init__(self, knn_model: cv2_ml_KNearest, sift_db: SIFTDatabase):
        self.__knn_model = knn_model
        self.__sift_db = sift_db

    @property
    def knn_model(self):
        if not self.__knn_model:
            raise ValueError("`knn_model` unset.")
        return self.__knn_model

    @knn_model.setter
    def knn_model(self, value: cv2_ml_KNearest):
        self.__knn_model = value

    @property
    def sift_db(self):
        if not self.__sift_db:
            raise ValueError("`sift_db` unset.")
        return self.__sift_db

    @sift_db.setter
    def sift_db(self, value: SIFTDatabase):
        self.__sift_db = value

    @lru_cache
    def _get_digit_widths(self, num_list: Sequence[int], factor: float):
        widths = set()
        for n in num_list:
            lower = math.floor(n * factor)
            upper = math.ceil(n * factor)
            widths.update(range(lower, upper + 1))
        return widths

    def _base_ocr_pfl(self, roi_masked: Mat, factor: float = 1.0):
        contours, _ = cv2.findContours(
            roi_masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        filtered_contours = [c for c in contours if cv2.contourArea(c) >= 5 * factor]
        rects = [cv2.boundingRect(c) for c in filtered_contours]
        rects = FixRects.connect_broken(rects, roi_masked.shape[1], roi_masked.shape[0])
        rect_contour_map = dict(zip(rects, filtered_contours))

        filtered_rects = [r for r in rects if r[2] >= 5 * factor and r[3] >= 6 * factor]
        filtered_rects = FixRects.split_connected(roi_masked, filtered_rects)
        filtered_rects = sorted(filtered_rects, key=lambda r: r[0])

        roi_ocr = roi_masked.copy()
        filtered_contours_flattened = {tuple(c.flatten()) for c in filtered_contours}
        for contour in contours:
            if tuple(contour.flatten()) in filtered_contours_flattened:
                continue
            roi_ocr = cv2.fillPoly(roi_ocr, [contour], [0])
        digit_rois = [
            resize_fill_square(crop_xywh(roi_ocr, r), 20)
            for r in sorted(filtered_rects, key=lambda r: r[0])
        ]
        # [cv2.imshow(f"r{i}", r) for i, r in enumerate(digit_rois)]
        # cv2.waitKey(0)
        samples = preprocess_hog(digit_rois)
        return ocr_digit_samples_knn(samples, self.knn_model)

    def ocr_song_id(self, rois: DeviceV2Rois):
        cover = cv2.cvtColor(rois.cover, cv2.COLOR_BGR2GRAY)
        return self.sift_db.lookup_img(cover)[0]

    def ocr_rating_class(self, rois: DeviceV2Rois):
        roi = cv2.cvtColor(rois.max_recall_rating_class, cv2.COLOR_BGR2HSV)
        results = [mask_pst(roi), mask_prs(roi), mask_ftr(roi), mask_byd(roi)]
        return max(enumerate(results), key=lambda i: np.count_nonzero(i[1]))[0]

    def ocr_score(self, rois: DeviceV2Rois):
        roi = cv2.cvtColor(rois.score, cv2.COLOR_BGR2HSV)
        roi = mask_white(roi)
        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if h < roi.shape[0] * 0.6:
                roi = cv2.fillPoly(roi, [contour], [0])
        return ocr_digits_by_contour_knn(roi, self.knn_model)

    def ocr_pure(self, rois: DeviceV2Rois):
        roi = mask_gray(rois.pure)
        return self._base_ocr_pfl(roi, rois.sizes.factor)

    def ocr_far(self, rois: DeviceV2Rois):
        roi = mask_gray(rois.far)
        return self._base_ocr_pfl(roi, rois.sizes.factor)

    def ocr_lost(self, rois: DeviceV2Rois):
        roi = mask_gray(rois.lost)
        return self._base_ocr_pfl(roi, rois.sizes.factor)

    def ocr_max_recall(self, rois: DeviceV2Rois):
        roi = mask_gray(rois.max_recall_rating_class)
        roi_closed = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, MAX_RECALL_CLOSE_KERNEL)
        contours, _ = cv2.findContours(
            roi_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        rects = sorted(
            [cv2.boundingRect(c) for c in contours], key=lambda r: r[0], reverse=True
        )
        max_recall_roi = crop_xywh(roi, rects[0])
        return ocr_digits_by_contour_knn(max_recall_roi, self.knn_model)

    def ocr(self, rois: DeviceV2Rois):
        song_id = self.ocr_song_id(rois)
        rating_class = self.ocr_rating_class(rois)
        score = self.ocr_score(rois)
        pure = self.ocr_pure(rois)
        far = self.ocr_far(rois)
        lost = self.ocr_lost(rois)
        max_recall = self.ocr_max_recall(rois)

        return DeviceOcrResult(
            rating_class=rating_class,
            pure=pure,
            far=far,
            lost=lost,
            score=score,
            max_recall=max_recall,
            song_id=song_id,
        )
