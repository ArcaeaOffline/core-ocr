from typing import List

import cv2

from ...crop import crop_xywh
from ...mask import mask_gray, mask_white
from ...ocr import ocr_digits_by_contour_knn, ocr_rating_class
from ...types import Mat, cv2_ml_KNearest
from ..shared import DeviceOcrResult
from .crop import *
from .definition import DeviceV1


class DeviceV1Ocr:
    def __init__(self, device: DeviceV1, knn_model: cv2_ml_KNearest):
        self.__device = device
        self.__knn_model = knn_model

    @property
    def device(self):
        return self.__device

    @device.setter
    def device(self, value):
        self.__device = value

    @property
    def knn_model(self):
        return self.__knn_model

    @knn_model.setter
    def knn_model(self, value):
        self.__knn_model = value

    def preprocess_score_roi(self, __roi_gray: Mat) -> List[Mat]:
        roi_gray = __roi_gray.copy()
        contours, _ = cv2.findContours(
            roi_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        for contour in contours:
            rect = cv2.boundingRect(contour)
            if rect[3] > roi_gray.shape[0] * 0.6:
                continue
            roi_gray = cv2.fillPoly(roi_gray, [contour], 0)
        return roi_gray

    def ocr(self, img_bgr: Mat):
        rating_class_roi = crop_to_rating_class(img_bgr, self.device)
        rating_class = ocr_rating_class(rating_class_roi)

        pfl_mr_roi = [
            crop_to_pure(img_bgr, self.device),
            crop_to_far(img_bgr, self.device),
            crop_to_lost(img_bgr, self.device),
            crop_to_max_recall(img_bgr, self.device),
        ]
        pfl_mr_roi = [mask_gray(roi) for roi in pfl_mr_roi]

        pure, far, lost = [
            ocr_digits_by_contour_knn(roi, self.knn_model) for roi in pfl_mr_roi[:3]
        ]

        max_recall_contours, _ = cv2.findContours(
            pfl_mr_roi[3], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        max_recall_rects = [cv2.boundingRect(c) for c in max_recall_contours]
        max_recall_rect = sorted(max_recall_rects, key=lambda r: r[0])[-1]
        max_recall_roi = crop_xywh(img_bgr, max_recall_rect)
        max_recall = ocr_digits_by_contour_knn(max_recall_roi, self.knn_model)

        score_roi = crop_to_score(img_bgr, self.device)
        score_roi = mask_white(score_roi)
        score_roi = self.preprocess_score_roi(score_roi)
        score = ocr_digits_by_contour_knn(score_roi, self.knn_model)

        return DeviceOcrResult(
            song_id=None,
            title=None,
            rating_class=rating_class,
            pure=pure,
            far=far,
            lost=lost,
            score=score,
            max_recall=max_recall,
            clear_type=None,
        )
