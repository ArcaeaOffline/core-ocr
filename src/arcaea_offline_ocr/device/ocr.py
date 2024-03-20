import cv2
import numpy as np

from ..crop import crop_xywh
from ..ocr import (
    FixRects,
    ocr_digit_samples_knn,
    ocr_digits_by_contour_knn,
    preprocess_hog,
    resize_fill_square,
)
from ..phash_db import ImagePhashDatabase
from ..types import Mat
from .common import DeviceOcrResult
from .rois.extractor import DeviceRoisExtractor
from .rois.masker import DeviceRoisMasker


class DeviceOcr:
    def __init__(
        self,
        extractor: DeviceRoisExtractor,
        masker: DeviceRoisMasker,
        knn_model: cv2.ml.KNearest,
        phash_db: ImagePhashDatabase,
    ):
        self.extractor = extractor
        self.masker = masker
        self.knn_model = knn_model
        self.phash_db = phash_db

    def pfl(self, roi_gray: Mat, factor: float = 1.25):
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
            resize_fill_square(crop_xywh(roi_ocr, r), 20) for r in filtered_rects
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
            if (
                cv2.boundingRect(contour)[3] < roi.shape[0] * 0.6
            ):  # h < score_component_h * 0.6
                roi = cv2.fillPoly(roi, [contour], [0])
        return ocr_digits_by_contour_knn(roi, self.knn_model)

    def rating_class(self):
        roi = self.extractor.rating_class
        results = [
            self.masker.rating_class_pst(roi),
            self.masker.rating_class_prs(roi),
            self.masker.rating_class_ftr(roi),
            self.masker.rating_class_byd(roi),
            self.masker.rating_class_etr(roi),
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

    def lookup_song_id(self):
        return self.phash_db.lookup_jacket(
            cv2.cvtColor(self.extractor.jacket, cv2.COLOR_BGR2GRAY)
        )

    def song_id(self):
        return self.lookup_song_id()[0]

    @staticmethod
    def preprocess_char_icon(img_gray: Mat):
        h, w = img_gray.shape[:2]
        img = cv2.copyMakeBorder(img_gray, w - h, 0, 0, 0, cv2.BORDER_REPLICATE)
        h, w = img.shape[:2]
        img = cv2.fillPoly(
            img,
            [
                np.array([[0, 0], [round(w / 2), 0], [0, round(h / 2)]], np.int32),
                np.array([[w, 0], [round(w / 2), 0], [w, round(h / 2)]], np.int32),
                np.array([[0, h], [round(w / 2), h], [0, round(h / 2)]], np.int32),
                np.array([[w, h], [round(w / 2), h], [w, round(h / 2)]], np.int32),
            ],
            (128),
        )
        return img

    def lookup_partner_id(self):
        return self.phash_db.lookup_partner_icon(
            self.preprocess_char_icon(
                cv2.cvtColor(self.extractor.partner_icon, cv2.COLOR_BGR2GRAY)
            )
        )

    def partner_id(self):
        return self.lookup_partner_id()[0]

    def ocr(self) -> DeviceOcrResult:
        rating_class = self.rating_class()
        pure = self.pure()
        far = self.far()
        lost = self.lost()
        score = self.score()
        max_recall = self.max_recall()
        clear_status = self.clear_status()

        hash_len = self.phash_db.hash_size**2
        song_id, song_id_distance = self.lookup_song_id()
        partner_id, partner_id_distance = self.lookup_partner_id()

        return DeviceOcrResult(
            rating_class=rating_class,
            pure=pure,
            far=far,
            lost=lost,
            score=score,
            max_recall=max_recall,
            song_id=song_id,
            song_id_possibility=1 - song_id_distance / hash_len,
            clear_status=clear_status,
            partner_id=partner_id,
            partner_id_possibility=1 - partner_id_distance / hash_len,
        )
