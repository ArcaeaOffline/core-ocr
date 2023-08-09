from datetime import datetime
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import attrs
import cv2
import numpy as np

from ....crop import crop_xywh
from ....ocr import preprocess_hog
from ....types import Mat, XYWHRect, cv2_ml_KNearest
from ....utils import construct_int_xywh_rect
from .colors import *
from .rois import ChieriBotV4Rois

if TYPE_CHECKING:
    from paddleocr import PaddleOCR


@attrs.define
class ChieriBotV4OcrResultItem:
    rating_class: int
    title: str
    score: int
    pure: int
    far: int
    lost: int
    date: Union[datetime, str]


class ChieriBotV4Ocr:
    def __init__(
        self,
        paddle_ocr: "PaddleOCR",
        knn_digits_model: cv2_ml_KNearest,
        factor: Optional[float] = 1.0,
    ):
        self.__paddle_ocr = paddle_ocr
        self.__knn_digits_model = knn_digits_model
        self.__rois = ChieriBotV4Rois(factor)

    @property
    def paddle_ocr(self):
        return self.__paddle_ocr

    @paddle_ocr.setter
    def paddle_ocr(self, paddle_ocr: "PaddleOCR"):
        self.__paddle_ocr = paddle_ocr

    @property
    def knn_digits_model(self):
        return self.__knn_digits_model

    @knn_digits_model.setter
    def knn_digits_model(self, knn_digits_model: Mat):
        self.__knn_digits_model = knn_digits_model

    @property
    def rois(self):
        return self.__rois

    @property
    def factor(self):
        return self.__rois.factor

    @factor.setter
    def factor(self, factor: float):
        self.__rois.factor = factor

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

    def ocr_component_title(self, component_bgr: Mat) -> str:
        # sourcery skip: inline-immediately-returned-variable
        title_rect = construct_int_xywh_rect(self.rois.component_rois.title_rect)
        title_roi = crop_xywh(component_bgr, title_rect)
        ocr_result = self.paddle_ocr.ocr(title_roi, cls=False)
        title = ocr_result[0][-1][1][0] if ocr_result and ocr_result[0] else ""
        return title

    def ocr_component_score(self, component_bgr: Mat) -> int:
        # sourcery skip: inline-immediately-returned-variable
        score_rect = construct_int_xywh_rect(self.rois.component_rois.score_rect)
        score_roi = cv2.cvtColor(
            crop_xywh(component_bgr, score_rect), cv2.COLOR_BGR2GRAY
        )
        _, score_roi = cv2.threshold(
            score_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        score_str = self.paddle_ocr.ocr(score_roi, cls=False)[0][-1][1][0]
        score = int(score_str.replace("'", "").replace(" ", ""))
        return score

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

    def ocr_component_pfl(self, component_bgr: Mat) -> Tuple[int, int, int]:
        try:
            pfl_roi = self.preprocess_component_pfl(component_bgr)
            pfl_rects = self.find_pfl_rects(pfl_roi)
            pure_far_lost = []
            for pfl_roi_rect in pfl_rects:
                roi = crop_xywh(pfl_roi, pfl_roi_rect)
                digit_contours, _ = cv2.findContours(
                    roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                digit_rects = sorted(
                    [cv2.boundingRect(c) for c in digit_contours],
                    key=lambda r: r[0],
                )
                digits = []
                for digit_rect in digit_rects:
                    digit = crop_xywh(roi, digit_rect)
                    digit = cv2.resize(digit, (20, 20))
                    digits.append(digit)
                samples = preprocess_hog(digits)

                _, results, _, _ = self.knn_digits_model.findNearest(samples, 4)
                results = [str(int(i)) for i in results.ravel()]
                pure_far_lost.append(int("".join(results)))
            return tuple(pure_far_lost)
        except Exception:
            return (-1, -1, -1)

    def ocr_component(self, component_bgr: Mat) -> ChieriBotV4OcrResultItem:
        component_blur = cv2.GaussianBlur(component_bgr, (5, 5), 0)
        rating_class = self.ocr_component_rating_class(component_blur)
        title = self.ocr_component_title(component_blur)
        score = self.ocr_component_score(component_blur)
        pure, far, lost = self.ocr_component_pfl(component_bgr)
        return ChieriBotV4OcrResultItem(
            rating_class=rating_class,
            title=title,
            score=score,
            pure=pure,
            far=far,
            lost=lost,
            date="",
        )

    def ocr(self, img_bgr: Mat) -> List[ChieriBotV4OcrResultItem]:
        self.factor = img_bgr.shape[0] / 4400
        return [
            self.ocr_component(component_bgr)
            for component_bgr in self.rois.components(img_bgr)
        ]