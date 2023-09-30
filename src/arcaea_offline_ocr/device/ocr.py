import cv2
import numpy as np
from PIL import Image

from ..crop import crop_xywh
from ..ocr import (
    FixRects,
    ocr_digit_samples_knn,
    ocr_digits_by_contour_knn,
    preprocess_hog,
    resize_fill_square,
)
from ..phash_db import ImagePHashDatabase
from .roi.extractor import DeviceRoiExtractor
from .roi.masker import DeviceRoiMasker


class DeviceOcr:
    def __init__(
        self,
        extractor: DeviceRoiExtractor,
        masker: DeviceRoiMasker,
        knn_model: cv2.ml.KNearest,
        phash_db: ImagePHashDatabase,
    ):
        self.extractor = extractor
        self.masker = masker
        self.knn_model = knn_model
        self.phash_db = phash_db

    def pfl(self, roi_gray: cv2.Mat, factor: float = 1.25):
        contours, _ = cv2.findContours(
            roi_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        filtered_contours = [c for c in contours if cv2.contourArea(c) >= 5 * factor]
        rects = [cv2.boundingRect(c) for c in filtered_contours]
        rects = FixRects.connect_broken(rects, roi_gray.shape[1], roi_gray.shape[0])

        filtered_rects = [r for r in rects if r[2] >= 5 * factor and r[3] >= 6 * factor]
        filtered_rects = FixRects.split_connected(roi_gray, filtered_rects)
        filtered_rects = sorted(filtered_rects, key=lambda r: r[0])

        roi_ocr = roi_gray.copy()
        filtered_contours_flattened = {tuple(c.flatten()) for c in filtered_contours}
        for contour in contours:
            if tuple(contour.flatten()) in filtered_contours_flattened:
                continue
            roi_ocr = cv2.fillPoly(roi_ocr, [contour], [0])
        digit_rois = [
            resize_fill_square(crop_xywh(roi_ocr, r), 20)
            for r in sorted(filtered_rects, key=lambda r: r[0])
        ]

        samples = preprocess_hog(digit_rois)
        return ocr_digit_samples_knn(samples, self.knn_model)

    def pure(self):
        return self.pfl(self.masker.pure(self.extractor.pure))

    def far(self):
        return self.pfl(self.masker.far(self.extractor.far))

    def lost(self):
        return self.pfl(self.masker.lost(self.extractor.lost))

    def score(self):
        roi = self.masker.score(self.extractor.score)
        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if h < roi.shape[0] * 0.6:
                roi = cv2.fillPoly(roi, [contour], [0])
        return ocr_digits_by_contour_knn(roi, self.knn_model)

    def rating_class(self):
        roi = self.extractor.rating_class
        results = [
            self.masker.rating_class_pst(roi),
            self.masker.rating_class_prs(roi),
            self.masker.rating_class_ftr(roi),
            self.masker.rating_class_byd(roi),
        ]
        return max(enumerate(results), key=lambda i: np.count_nonzero(i[1]))[0]

    def max_recall(self):
        return ocr_digits_by_contour_knn(
            self.masker.max_recall(self.extractor.max_recall), self.knn_model
        )

    def clear_status(self):
        roi = self.extractor.clear_status
        results = [
            self.masker.clear_status_track_lost(roi),
            self.masker.clear_status_track_complete(roi),
            self.masker.clear_status_full_recall(roi),
            self.masker.clear_status_pure_memory(roi),
        ]
        return max(enumerate(results), key=lambda i: np.count_nonzero(i[1]))[0]

    def song_id(self):
        return self.phash_db.lookup_image(Image.fromarray(self.extractor.jacket))[0]
