import cv2
import numpy as np


class RgbGrayMasker:
    @staticmethod
    def gray(
        roi_bgr: cv2.typing.MatLike,
        lower: cv2.typing.MatLike,
        upper: cv2.typing.MatLike,
    ) -> cv2.typing.MatLike:
        """
        Mask gray pixels: BGR values within ``[lower, upper]`` whose channel
        spread (max - min) is at most 5. Pixels outside the spread are zeroed
        and the image eroded 2x2 before the range check.
        """
        bgr_value_equal_mask = np.max(roi_bgr, axis=2) - np.min(roi_bgr, axis=2) <= 5
        img_bgr = roi_bgr.copy()
        img_bgr[~bgr_value_equal_mask] = np.array([0, 0, 0], roi_bgr.dtype)
        img_bgr = cv2.erode(img_bgr, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
        return cv2.inRange(img_bgr, lower, upper)
