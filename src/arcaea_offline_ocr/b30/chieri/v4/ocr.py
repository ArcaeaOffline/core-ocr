from math import floor
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ....crop import crop_xywh
from ....ocr import (
    FixRects,
    ocr_digits_by_contour_knn,
    preprocess_hog,
    resize_fill_square,
)
from ....phash_db import ImagePhashDatabase
from ....types import Mat
from ....utils import construct_int_xywh_rect
from ...shared import B30OcrResultItem
from .colors import *
from .rois import ChieriBotV4Rois


class ChieriBotV4Ocr:
    def __init__(
        self,
        score_knn: cv2.ml.KNearest,
        pfl_knn: cv2.ml.KNearest,
        phash_db: ImagePhashDatabase,
        factor: Optional[float] = 1.0,
    ):
        self.__score_knn = score_knn
        self.__pfl_knn = pfl_knn
        self.__phash_db = phash_db
        self.__rois = ChieriBotV4Rois(factor)

    @property
    def score_knn(self):
        return self.__score_knn

    @score_knn.setter
    def score_knn(self, knn_digits_model: cv2.ml.KNearest):
        self.__score_knn = knn_digits_model

    @property
    def pfl_knn(self):
        return self.__pfl_knn

    @pfl_knn.setter
    def pfl_knn(self, knn_digits_model: cv2.ml.KNearest):
        self.__pfl_knn = knn_digits_model

    @property
    def phash_db(self):
        return self.__phash_db

    @phash_db.setter
    def phash_db(self, phash_db: ImagePhashDatabase):
        self.__phash_db = phash_db

    @property
    def rois(self):
        return self.__rois

    @property
    def factor(self):
        return self.__rois.factor

    @factor.setter
    def factor(self, factor: float):
        self.__rois.factor = factor

    def set_factor(self, img: Mat):
        self.factor = img.shape[0] / 4400

    def ocr_component_rating_class(self, component_bgr: Mat) -> int:
        rating_class_rect = construct_int_xywh_rect(
            self.rois.component_rois.rating_class_rect
        )
        rating_class_roi = crop_xywh(component_bgr, rating_class_rect)
        rating_class_roi = cv2.cvtColor(rating_class_roi, cv2.COLOR_BGR2HSV)
        rating_class_masks = [
            cv2.inRange(rating_class_roi, PRS_MIN_HSV, PRS_MAX_HSV),
            cv2.inRange(rating_class_roi, FTR_MIN_HSV, FTR_MAX_HSV),
            cv2.inRange(rating_class_roi, BYD_MIN_HSV, BYD_MAX_HSV),
        ]  # prs, ftr, byd only
        rating_class_results = [np.count_nonzero(m) for m in rating_class_masks]
        if max(rating_class_results) < 70:
            return 0
        else:
            return max(enumerate(rating_class_results), key=lambda i: i[1])[0] + 1

    def ocr_component_song_id(self, component_bgr: Mat):
        jacket_rect = construct_int_xywh_rect(
            self.rois.component_rois.jacket_rect, floor
        )
        jacket_roi = cv2.cvtColor(
            crop_xywh(component_bgr, jacket_rect), cv2.COLOR_BGR2GRAY
        )
        return self.phash_db.lookup_jacket(jacket_roi)[0]

    def ocr_component_score_knn(self, component_bgr: Mat) -> int:
        # sourcery skip: inline-immediately-returned-variable
        score_rect = construct_int_xywh_rect(self.rois.component_rois.score_rect)
        score_roi = cv2.cvtColor(
            crop_xywh(component_bgr, score_rect), cv2.COLOR_BGR2GRAY
        )
        _, score_roi = cv2.threshold(
            score_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        if score_roi[1][1] == 255:
            score_roi = 255 - score_roi

        contours, _ = cv2.findContours(
            score_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for contour in contours:
            rect = cv2.boundingRect(contour)
            if rect[3] > score_roi.shape[0] * 0.5:
                continue
            score_roi = cv2.fillPoly(score_roi, [contour], 0)
        return ocr_digits_by_contour_knn(score_roi, self.score_knn)

    def find_pfl_rects(self, component_pfl_processed: Mat) -> List[List[int]]:
        # sourcery skip: inline-immediately-returned-variable
        pfl_roi_find = cv2.morphologyEx(
            component_pfl_processed,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, [10, 1]),
        )
        pfl_contours, _ = cv2.findContours(
            pfl_roi_find, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        pfl_rects = [cv2.boundingRect(c) for c in pfl_contours]
        pfl_rects = [
            r for r in pfl_rects if r[3] > component_pfl_processed.shape[0] * 0.1
        ]
        pfl_rects = sorted(pfl_rects, key=lambda r: r[1])
        pfl_rects_adjusted = [
            (
                max(rect[0] - 2, 0),
                rect[1],
                min(rect[2] + 2, component_pfl_processed.shape[1]),
                rect[3],
            )
            for rect in pfl_rects
        ]
        return pfl_rects_adjusted

    def preprocess_component_pfl(self, component_bgr: Mat) -> Mat:
        pfl_rect = construct_int_xywh_rect(self.rois.component_rois.pfl_rect)
        pfl_roi = crop_xywh(component_bgr, pfl_rect)
        pfl_roi_hsv = cv2.cvtColor(pfl_roi, cv2.COLOR_BGR2HSV)

        # fill the pfl bg with background color
        bg_point = [round(i) for i in self.rois.component_rois.bg_point]
        bg_color = component_bgr[bg_point[1]][bg_point[0]]
        pure_bg_mask = cv2.inRange(pfl_roi_hsv, PURE_BG_MIN_HSV, PURE_BG_MAX_HSV)
        far_bg_mask = cv2.inRange(pfl_roi_hsv, FAR_BG_MIN_HSV, FAR_BG_MAX_HSV)
        lost_bg_mask = cv2.inRange(pfl_roi_hsv, LOST_BG_MIN_HSV, LOST_BG_MAX_HSV)
        pfl_roi[np.where(pure_bg_mask != 0)] = bg_color
        pfl_roi[np.where(far_bg_mask != 0)] = bg_color
        pfl_roi[np.where(lost_bg_mask != 0)] = bg_color

        # threshold
        pfl_roi = cv2.cvtColor(pfl_roi, cv2.COLOR_BGR2GRAY)
        # get threshold of blurred image, try ignoring the lines of bg bar
        pfl_roi_blurred = cv2.GaussianBlur(pfl_roi, (5, 5), 0)
        # pfl_roi_blurred = cv2.medianBlur(pfl_roi, 3)
        _, pfl_roi_blurred_threshold = cv2.threshold(
            pfl_roi_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        # and a threshold of the original roi
        _, pfl_roi_threshold = cv2.threshold(
            pfl_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        # turn thresholds into black background
        if pfl_roi_blurred_threshold[2][2] == 255:
            pfl_roi_blurred_threshold = 255 - pfl_roi_blurred_threshold
        if pfl_roi_threshold[2][2] == 255:
            pfl_roi_threshold = 255 - pfl_roi_threshold
        # return a bitwise_and result
        result = cv2.bitwise_and(pfl_roi_blurred_threshold, pfl_roi_threshold)
        result_eroded = cv2.erode(
            result, cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2))
        )
        return result_eroded if len(self.find_pfl_rects(result_eroded)) == 3 else result

    def ocr_component_pfl(
        self, component_bgr: Mat
    ) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        try:
            pfl_roi = self.preprocess_component_pfl(component_bgr)
            pfl_rects = self.find_pfl_rects(pfl_roi)
            pure_far_lost = []
            for pfl_roi_rect in pfl_rects:
                roi = crop_xywh(pfl_roi, pfl_roi_rect)
                digit_contours, _ = cv2.findContours(
                    roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                digit_rects = [cv2.boundingRect(c) for c in digit_contours]
                digit_rects = FixRects.connect_broken(
                    digit_rects, roi.shape[1], roi.shape[0]
                )
                digit_rects = FixRects.split_connected(roi, digit_rects)
                digit_rects = sorted(digit_rects, key=lambda r: r[0])
                digits = []
                for digit_rect in digit_rects:
                    digit = crop_xywh(roi, digit_rect)
                    digit = resize_fill_square(digit, 20)
                    digits.append(digit)
                samples = preprocess_hog(digits)

                _, results, _, _ = self.pfl_knn.findNearest(samples, 4)
                results = [str(int(i)) for i in results.ravel()]
                pure_far_lost.append(int("".join(results)))
            return tuple(pure_far_lost)
        except Exception:
            return (None, None, None)

    def ocr_component(self, component_bgr: Mat) -> B30OcrResultItem:
        component_blur = cv2.GaussianBlur(component_bgr, (5, 5), 0)
        rating_class = self.ocr_component_rating_class(component_blur)
        song_id = self.ocr_component_song_id(component_bgr)
        # title = self.ocr_component_title(component_blur)
        # score = self.ocr_component_score(component_blur)
        score = self.ocr_component_score_knn(component_bgr)
        pure, far, lost = self.ocr_component_pfl(component_bgr)
        return B30OcrResultItem(
            song_id=song_id,
            rating_class=rating_class,
            # title=title,
            score=score,
            pure=pure,
            far=far,
            lost=lost,
            date=None,
        )

    def ocr(self, img_bgr: Mat) -> List[B30OcrResultItem]:
        self.set_factor(img_bgr)
        return [
            self.ocr_component(component_bgr)
            for component_bgr in self.rois.components(img_bgr)
        ]
