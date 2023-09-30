import cv2
import numpy as np

from .common import DeviceAutoRoiMasker

GRAY_BGR_MIN = np.array([50] * 3, np.uint8)
GRAY_BGR_MAX = np.array([160] * 3, np.uint8)

WHITE_HSV_MIN = np.array([0, 0, 240], np.uint8)
WHITE_HSV_MAX = np.array([179, 10, 255], np.uint8)

PST_HSV_MIN = np.array([100, 50, 80], np.uint8)
PST_HSV_MAX = np.array([100, 255, 255], np.uint8)

PRS_HSV_MIN = np.array([43, 40, 75], np.uint8)
PRS_HSV_MAX = np.array([50, 155, 190], np.uint8)

FTR_HSV_MIN = np.array([149, 30, 0], np.uint8)
FTR_HSV_MAX = np.array([155, 181, 150], np.uint8)

BYD_HSV_MIN = np.array([170, 50, 50], np.uint8)
BYD_HSV_MAX = np.array([179, 210, 198], np.uint8)

TRACK_LOST_HSV_MIN = np.array([170, 75, 90], np.uint8)
TRACK_LOST_HSV_MAX = np.array([175, 170, 160], np.uint8)

TRACK_COMPLETE_HSV_MIN = np.array([140, 0, 50], np.uint8)
TRACK_COMPLETE_HSV_MAX = np.array([145, 50, 130], np.uint8)

FULL_RECALL_HSV_MIN = np.array([140, 60, 80], np.uint8)
FULL_RECALL_HSV_MAX = np.array([150, 130, 145], np.uint8)

PURE_MEMORY_HSV_MIN = np.array([90, 70, 80], np.uint8)
PURE_MEMORY_HSV_MAX = np.array([110, 200, 175], np.uint8)


class DeviceAutoRoiMaskerT1(DeviceAutoRoiMasker):
    @classmethod
    def gray(cls, roi_bgr: cv2.Mat) -> cv2.Mat:
        bgr_value_equal_mask = np.max(roi_bgr, axis=2) - np.min(roi_bgr, axis=2) <= 5
        img_bgr = roi_bgr.copy()
        img_bgr[~bgr_value_equal_mask] = np.array([0, 0, 0], roi_bgr.dtype)
        img_bgr = cv2.erode(img_bgr, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
        img_bgr = cv2.dilate(img_bgr, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1)))
        return cv2.inRange(img_bgr, GRAY_BGR_MIN, GRAY_BGR_MAX)

    @classmethod
    def pure(cls, roi_bgr: cv2.Mat) -> cv2.Mat:
        return cls.gray(roi_bgr)

    @classmethod
    def far(cls, roi_bgr: cv2.Mat) -> cv2.Mat:
        return cls.gray(roi_bgr)

    @classmethod
    def lost(cls, roi_bgr: cv2.Mat) -> cv2.Mat:
        return cls.gray(roi_bgr)

    @classmethod
    def score(cls, roi_bgr: cv2.Mat) -> cv2.Mat:
        return cv2.inRange(
            cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV), WHITE_HSV_MIN, WHITE_HSV_MAX
        )

    @classmethod
    def rating_class_pst(cls, roi_bgr: cv2.Mat) -> cv2.Mat:
        return cv2.inRange(
            cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV), PST_HSV_MIN, PST_HSV_MAX
        )

    @classmethod
    def rating_class_prs(cls, roi_bgr: cv2.Mat) -> cv2.Mat:
        return cv2.inRange(
            cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV), PRS_HSV_MIN, PRS_HSV_MAX
        )

    @classmethod
    def rating_class_ftr(cls, roi_bgr: cv2.Mat) -> cv2.Mat:
        return cv2.inRange(
            cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV), FTR_HSV_MIN, FTR_HSV_MAX
        )

    @classmethod
    def rating_class_byd(cls, roi_bgr: cv2.Mat) -> cv2.Mat:
        return cv2.inRange(
            cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV), BYD_HSV_MIN, BYD_HSV_MAX
        )

    @classmethod
    def max_recall(cls, roi_bgr: cv2.Mat) -> cv2.Mat:
        return cls.gray(roi_bgr)

    @classmethod
    def clear_status_track_lost(cls, roi_bgr: cv2.Mat) -> cv2.Mat:
        return cv2.inRange(
            cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV),
            TRACK_LOST_HSV_MIN,
            TRACK_LOST_HSV_MAX,
        )

    @classmethod
    def clear_status_track_complete(cls, roi_bgr: cv2.Mat) -> cv2.Mat:
        return cv2.inRange(
            cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV),
            TRACK_COMPLETE_HSV_MIN,
            TRACK_COMPLETE_HSV_MAX,
        )

    @classmethod
    def clear_status_full_recall(cls, roi_bgr: cv2.Mat) -> cv2.Mat:
        return cv2.inRange(
            cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV),
            FULL_RECALL_HSV_MIN,
            FULL_RECALL_HSV_MAX,
        )

    @classmethod
    def clear_status_pure_memory(cls, roi_bgr: cv2.Mat) -> cv2.Mat:
        return cv2.inRange(
            cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV),
            PURE_MEMORY_HSV_MIN,
            PURE_MEMORY_HSV_MAX,
        )
