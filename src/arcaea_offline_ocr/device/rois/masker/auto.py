import cv2
import numpy as np

from ....types import Mat
from .common import DeviceRoisMasker


class DeviceRoisMaskerAuto(DeviceRoisMasker):
    # pylint: disable=abstract-method

    @staticmethod
    def mask_bgr_in_hsv(roi_bgr: Mat, hsv_lower: Mat, hsv_upper: Mat):
        return cv2.inRange(
            cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV), hsv_lower, hsv_upper
        )


class DeviceRoisMaskerAutoT1(DeviceRoisMaskerAuto):
    GRAY_BGR_MIN = np.array([50] * 3, np.uint8)
    GRAY_BGR_MAX = np.array([160] * 3, np.uint8)

    SCORE_HSV_MIN = np.array([0, 0, 180], np.uint8)
    SCORE_HSV_MAX = np.array([179, 255, 255], np.uint8)

    PST_HSV_MIN = np.array([100, 50, 80], np.uint8)
    PST_HSV_MAX = np.array([100, 255, 255], np.uint8)

    PRS_HSV_MIN = np.array([43, 40, 75], np.uint8)
    PRS_HSV_MAX = np.array([50, 155, 190], np.uint8)

    FTR_HSV_MIN = np.array([149, 30, 0], np.uint8)
    FTR_HSV_MAX = np.array([155, 181, 150], np.uint8)

    BYD_HSV_MIN = np.array([170, 50, 50], np.uint8)
    BYD_HSV_MAX = np.array([179, 210, 198], np.uint8)

    ETR_HSV_MIN = np.array([130, 60, 80], np.uint8)
    ETR_HSV_MAX = np.array([140, 145, 180], np.uint8)

    TRACK_LOST_HSV_MIN = np.array([170, 75, 90], np.uint8)
    TRACK_LOST_HSV_MAX = np.array([175, 170, 160], np.uint8)

    TRACK_COMPLETE_HSV_MIN = np.array([140, 0, 50], np.uint8)
    TRACK_COMPLETE_HSV_MAX = np.array([145, 50, 130], np.uint8)

    FULL_RECALL_HSV_MIN = np.array([140, 60, 80], np.uint8)
    FULL_RECALL_HSV_MAX = np.array([150, 130, 145], np.uint8)

    PURE_MEMORY_HSV_MIN = np.array([90, 70, 80], np.uint8)
    PURE_MEMORY_HSV_MAX = np.array([110, 200, 175], np.uint8)

    @classmethod
    def gray(cls, roi_bgr: Mat) -> Mat:
        bgr_value_equal_mask = np.max(roi_bgr, axis=2) - np.min(roi_bgr, axis=2) <= 5
        img_bgr = roi_bgr.copy()
        img_bgr[~bgr_value_equal_mask] = np.array([0, 0, 0], roi_bgr.dtype)
        img_bgr = cv2.erode(img_bgr, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
        img_bgr = cv2.dilate(img_bgr, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1)))
        return cv2.inRange(img_bgr, cls.GRAY_BGR_MIN, cls.GRAY_BGR_MAX)

    @classmethod
    def pure(cls, roi_bgr: Mat) -> Mat:
        return cls.gray(roi_bgr)

    @classmethod
    def far(cls, roi_bgr: Mat) -> Mat:
        return cls.gray(roi_bgr)

    @classmethod
    def lost(cls, roi_bgr: Mat) -> Mat:
        return cls.gray(roi_bgr)

    @classmethod
    def score(cls, roi_bgr: Mat) -> Mat:
        return cls.mask_bgr_in_hsv(roi_bgr, cls.SCORE_HSV_MIN, cls.SCORE_HSV_MAX)

    @classmethod
    def rating_class_pst(cls, roi_bgr: Mat) -> Mat:
        return cls.mask_bgr_in_hsv(roi_bgr, cls.PST_HSV_MIN, cls.PST_HSV_MAX)

    @classmethod
    def rating_class_prs(cls, roi_bgr: Mat) -> Mat:
        return cls.mask_bgr_in_hsv(roi_bgr, cls.PRS_HSV_MIN, cls.PRS_HSV_MAX)

    @classmethod
    def rating_class_ftr(cls, roi_bgr: Mat) -> Mat:
        return cls.mask_bgr_in_hsv(roi_bgr, cls.FTR_HSV_MIN, cls.FTR_HSV_MAX)

    @classmethod
    def rating_class_byd(cls, roi_bgr: Mat) -> Mat:
        return cls.mask_bgr_in_hsv(roi_bgr, cls.BYD_HSV_MIN, cls.BYD_HSV_MAX)

    @classmethod
    def rating_class_etr(cls, roi_bgr: Mat) -> Mat:
        return cls.mask_bgr_in_hsv(roi_bgr, cls.ETR_HSV_MIN, cls.ETR_HSV_MAX)

    @classmethod
    def max_recall(cls, roi_bgr: Mat) -> Mat:
        return cls.gray(roi_bgr)

    @classmethod
    def clear_status_track_lost(cls, roi_bgr: Mat) -> Mat:
        return cls.mask_bgr_in_hsv(
            roi_bgr, cls.TRACK_LOST_HSV_MIN, cls.TRACK_LOST_HSV_MAX
        )

    @classmethod
    def clear_status_track_complete(cls, roi_bgr: Mat) -> Mat:
        return cls.mask_bgr_in_hsv(
            roi_bgr, cls.TRACK_COMPLETE_HSV_MIN, cls.TRACK_COMPLETE_HSV_MAX
        )

    @classmethod
    def clear_status_full_recall(cls, roi_bgr: Mat) -> Mat:
        return cls.mask_bgr_in_hsv(
            roi_bgr, cls.FULL_RECALL_HSV_MIN, cls.FULL_RECALL_HSV_MAX
        )

    @classmethod
    def clear_status_pure_memory(cls, roi_bgr: Mat) -> Mat:
        return cls.mask_bgr_in_hsv(
            roi_bgr, cls.PURE_MEMORY_HSV_MIN, cls.PURE_MEMORY_HSV_MAX
        )


class DeviceRoisMaskerAutoT2(DeviceRoisMaskerAuto):
    PFL_HSV_MIN = np.array([0, 0, 248], np.uint8)
    PFL_HSV_MAX = np.array([179, 10, 255], np.uint8)

    SCORE_HSV_MIN = np.array([0, 0, 180], np.uint8)
    SCORE_HSV_MAX = np.array([179, 255, 255], np.uint8)

    PST_HSV_MIN = np.array([100, 50, 80], np.uint8)
    PST_HSV_MAX = np.array([100, 255, 255], np.uint8)

    PRS_HSV_MIN = np.array([43, 40, 75], np.uint8)
    PRS_HSV_MAX = np.array([50, 155, 190], np.uint8)

    FTR_HSV_MIN = np.array([149, 30, 0], np.uint8)
    FTR_HSV_MAX = np.array([155, 181, 150], np.uint8)

    BYD_HSV_MIN = np.array([170, 50, 50], np.uint8)
    BYD_HSV_MAX = np.array([179, 210, 198], np.uint8)

    ETR_HSV_MIN = np.array([130, 60, 80], np.uint8)
    ETR_HSV_MAX = np.array([140, 145, 180], np.uint8)

    MAX_RECALL_HSV_MIN = np.array([125, 0, 0], np.uint8)
    MAX_RECALL_HSV_MAX = np.array([145, 100, 150], np.uint8)

    TRACK_LOST_HSV_MIN = np.array([170, 75, 90], np.uint8)
    TRACK_LOST_HSV_MAX = np.array([175, 170, 160], np.uint8)

    TRACK_COMPLETE_HSV_MIN = np.array([140, 0, 50], np.uint8)
    TRACK_COMPLETE_HSV_MAX = np.array([145, 50, 130], np.uint8)

    FULL_RECALL_HSV_MIN = np.array([140, 60, 80], np.uint8)
    FULL_RECALL_HSV_MAX = np.array([150, 130, 145], np.uint8)

    PURE_MEMORY_HSV_MIN = np.array([90, 70, 80], np.uint8)
    PURE_MEMORY_HSV_MAX = np.array([110, 200, 175], np.uint8)

    @classmethod
    def pfl(cls, roi_bgr: Mat) -> Mat:
        return cls.mask_bgr_in_hsv(roi_bgr, cls.PFL_HSV_MIN, cls.PFL_HSV_MAX)

    @classmethod
    def pure(cls, roi_bgr: Mat) -> Mat:
        return cls.pfl(roi_bgr)

    @classmethod
    def far(cls, roi_bgr: Mat) -> Mat:
        return cls.pfl(roi_bgr)

    @classmethod
    def lost(cls, roi_bgr: Mat) -> Mat:
        return cls.pfl(roi_bgr)

    @classmethod
    def score(cls, roi_bgr: Mat) -> Mat:
        return cls.mask_bgr_in_hsv(roi_bgr, cls.SCORE_HSV_MIN, cls.SCORE_HSV_MAX)

    @classmethod
    def rating_class_pst(cls, roi_bgr: Mat) -> Mat:
        return cls.mask_bgr_in_hsv(roi_bgr, cls.PST_HSV_MIN, cls.PST_HSV_MAX)

    @classmethod
    def rating_class_prs(cls, roi_bgr: Mat) -> Mat:
        return cls.mask_bgr_in_hsv(roi_bgr, cls.PRS_HSV_MIN, cls.PRS_HSV_MAX)

    @classmethod
    def rating_class_ftr(cls, roi_bgr: Mat) -> Mat:
        return cls.mask_bgr_in_hsv(roi_bgr, cls.FTR_HSV_MIN, cls.FTR_HSV_MAX)

    @classmethod
    def rating_class_byd(cls, roi_bgr: Mat) -> Mat:
        return cls.mask_bgr_in_hsv(roi_bgr, cls.BYD_HSV_MIN, cls.BYD_HSV_MAX)

    @classmethod
    def rating_class_etr(cls, roi_bgr: Mat) -> Mat:
        return cls.mask_bgr_in_hsv(roi_bgr, cls.ETR_HSV_MIN, cls.ETR_HSV_MAX)

    @classmethod
    def max_recall(cls, roi_bgr: Mat) -> Mat:
        return cls.mask_bgr_in_hsv(
            roi_bgr, cls.MAX_RECALL_HSV_MIN, cls.MAX_RECALL_HSV_MAX
        )

    @classmethod
    def clear_status_track_lost(cls, roi_bgr: Mat) -> Mat:
        return cls.mask_bgr_in_hsv(
            roi_bgr, cls.TRACK_LOST_HSV_MIN, cls.TRACK_LOST_HSV_MAX
        )

    @classmethod
    def clear_status_track_complete(cls, roi_bgr: Mat) -> Mat:
        return cls.mask_bgr_in_hsv(
            roi_bgr, cls.TRACK_COMPLETE_HSV_MIN, cls.TRACK_COMPLETE_HSV_MAX
        )

    @classmethod
    def clear_status_full_recall(cls, roi_bgr: Mat) -> Mat:
        return cls.mask_bgr_in_hsv(
            roi_bgr, cls.FULL_RECALL_HSV_MIN, cls.FULL_RECALL_HSV_MAX
        )

    @classmethod
    def clear_status_pure_memory(cls, roi_bgr: Mat) -> Mat:
        return cls.mask_bgr_in_hsv(
            roi_bgr, cls.PURE_MEMORY_HSV_MIN, cls.PURE_MEMORY_HSV_MAX
        )
