from typing import Optional

import attrs
import cv2
import numpy as np

from ...mask import mask_byd, mask_ftr, mask_gray, mask_prs, mask_pst, mask_white
from ...ocr import ocr_digits_knn_model
from ...types import Mat, cv2_ml_KNearest
from .find import find_digits
from .rois import DeviceV2Rois


@attrs.define
class DeviceV2OcrResult:
    pure: int
    far: int
    lost: int
    score: int
    rating_class: int
    max_recall: int
    title: Optional[str]


class DeviceV2Ocr:
    def __init__(self):
        self.__rois = None
        self.__knn_model = None

    @property
    def rois(self):
        if not self.__rois:
            raise ValueError("`rois` unset.")
        return self.__rois

    @rois.setter
    def rois(self, rois: DeviceV2Rois):
        self.__rois = rois

    @property
    def knn_model(self):
        if not self.__knn_model:
            raise ValueError("`knn_model` unset.")
        return self.__knn_model

    @knn_model.setter
    def knn_model(self, model: cv2_ml_KNearest):
        self.__knn_model = model

    def _base_ocr_digits(self, roi_processed: Mat):
        digits = find_digits(roi_processed)
        result = ""
        for digit in digits:
            roi_result = ocr_digits_knn_model(digit, self.knn_model)
            if roi_result is not None:
                result += str(roi_result)
        return int(result, base=10)

    @property
    def pure(self):
        roi = mask_gray(self.rois.pure)
        return self._base_ocr_digits(roi)

    @property
    def far(self):
        roi = mask_gray(self.rois.far)
        return self._base_ocr_digits(roi)

    @property
    def lost(self):
        roi = mask_gray(self.rois.lost)
        return self._base_ocr_digits(roi)

    @property
    def score(self):
        roi = cv2.cvtColor(self.rois.score, cv2.COLOR_BGR2HSV)
        roi = mask_white(roi)
        return self._base_ocr_digits(roi)

    @property
    def rating_class(self):
        roi = cv2.cvtColor(self.rois.max_recall_rating_class, cv2.COLOR_BGR2HSV)
        results = [
            mask_pst(roi),
            mask_prs(roi),
            mask_ftr(roi),
            mask_byd(roi),
        ]
        return max(enumerate(results), key=lambda e: np.count_nonzero(e[1]))[0]
