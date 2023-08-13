import cv2
import numpy as np

from ...crop import crop_xywh
from ...mask import mask_byd, mask_ftr, mask_gray, mask_prs, mask_pst, mask_white
from ...ocr import ocr_digits_by_contour_knn
from ...sift_db import SIFTDatabase
from ...types import Mat, cv2_ml_KNearest
from ..shared import DeviceOcrResult
from .find import find_digits_preprocess
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

    def _base_ocr_digits(self, roi_masked: Mat):
        return ocr_digits_by_contour_knn(
            find_digits_preprocess(roi_masked), self.knn_model
        )

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
        return self._base_ocr_digits(roi)

    def ocr_pure(self, rois: DeviceV2Rois):
        roi = mask_gray(rois.pure)
        return self._base_ocr_digits(roi)

    def ocr_far(self, rois: DeviceV2Rois):
        roi = mask_gray(rois.far)
        return self._base_ocr_digits(roi)

    def ocr_lost(self, rois: DeviceV2Rois):
        roi = mask_gray(rois.lost)
        return self._base_ocr_digits(roi)

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
        return self._base_ocr_digits(max_recall_roi)

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
