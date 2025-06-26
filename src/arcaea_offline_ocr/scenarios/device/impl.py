import cv2
import numpy as np

from arcaea_offline_ocr.providers import (
    ImageCategory,
    ImageIdProvider,
    OcrKNearestTextProvider,
)
from arcaea_offline_ocr.scenarios.base import OcrScenarioResult
from arcaea_offline_ocr.types import Mat

from .base import DeviceScenarioBase
from .extractor import DeviceRoisExtractor
from .masker import DeviceRoisMasker


class DeviceScenario(DeviceScenarioBase):
    def __init__(
        self,
        extractor: DeviceRoisExtractor,
        masker: DeviceRoisMasker,
        knn_provider: OcrKNearestTextProvider,
        image_id_provider: ImageIdProvider,
    ):
        self.extractor = extractor
        self.masker = masker
        self.knn_provider = knn_provider
        self.image_id_provider = image_id_provider

    def pfl(self, roi_gray: Mat, factor: float = 1.25):
        def contour_filter(cnt):
            return cv2.contourArea(cnt) >= 5 * factor

        contours = self.knn_provider.contours(roi_gray)
        contours_filtered = self.knn_provider.contours(
            roi_gray,
            contours_filter=contour_filter,
        )

        roi_ocr = roi_gray.copy()
        contours_filtered_flattened = {tuple(c.flatten()) for c in contours_filtered}
        for contour in contours:
            if tuple(contour.flatten()) in contours_filtered_flattened:
                continue
            roi_ocr = cv2.fillPoly(roi_ocr, [contour], [0])

        ocr_result = self.knn_provider.result(
            roi_ocr,
            contours_filter=lambda cnt: cv2.contourArea(cnt) >= 5 * factor,
            rects_filter=lambda rect: rect[2] >= 5 * factor and rect[3] >= 6 * factor,
        )

        return int(ocr_result) if ocr_result else 0

    def pure(self):
        return self.pfl(self.masker.pure(self.extractor.pure))

    def far(self):
        return self.pfl(self.masker.far(self.extractor.far))

    def lost(self):
        return self.pfl(self.masker.lost(self.extractor.lost))

    def score(self):
        roi = self.masker.score(self.extractor.score)
        contours = self.knn_provider.contours(roi)
        for contour in contours:
            if (
                cv2.boundingRect(contour)[3] < roi.shape[0] * 0.6
            ):  # h < score_component_h * 0.6
                roi = cv2.fillPoly(roi, [contour], [0])
        ocr_result = self.knn_provider.result(roi)
        return int(ocr_result) if ocr_result else 0

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
        ocr_result = self.knn_provider.result(
            self.masker.max_recall(self.extractor.max_recall),
        )
        return int(ocr_result) if ocr_result else None

    def clear_status(self):
        roi = self.extractor.clear_status
        results = [
            self.masker.clear_status_track_lost(roi),
            self.masker.clear_status_track_complete(roi),
            self.masker.clear_status_full_recall(roi),
            self.masker.clear_status_pure_memory(roi),
        ]
        return max(enumerate(results), key=lambda i: np.count_nonzero(i[1]))[0]

    def song_id_results(self):
        return self.image_id_provider.results(
            cv2.cvtColor(self.extractor.jacket, cv2.COLOR_BGR2GRAY),
            ImageCategory.JACKET,
        )

    @staticmethod
    def preprocess_char_icon(img_gray: Mat):
        h, w = img_gray.shape[:2]
        img = cv2.copyMakeBorder(img_gray, max(w - h, 0), 0, 0, 0, cv2.BORDER_REPLICATE)
        h, w = img.shape[:2]
        return cv2.fillPoly(
            img,
            [
                np.array([[0, 0], [round(w / 2), 0], [0, round(h / 2)]], np.int32),
                np.array([[w, 0], [round(w / 2), 0], [w, round(h / 2)]], np.int32),
                np.array([[0, h], [round(w / 2), h], [0, round(h / 2)]], np.int32),
                np.array([[w, h], [round(w / 2), h], [w, round(h / 2)]], np.int32),
            ],
            (128,),
        )

    def partner_id_results(self):
        return self.image_id_provider.results(
            self.preprocess_char_icon(
                cv2.cvtColor(self.extractor.partner_icon, cv2.COLOR_BGR2GRAY),
            ),
            ImageCategory.PARTNER_ICON,
        )

    def result(self):
        rating_class = self.rating_class()
        pure = self.pure()
        far = self.far()
        lost = self.lost()
        score = self.score()
        max_recall = self.max_recall()
        clear_status = self.clear_status()

        song_id_results = self.song_id_results()
        partner_id_results = self.partner_id_results()

        return OcrScenarioResult(
            song_id=song_id_results[0].image_id,
            song_id_results=song_id_results,
            rating_class=rating_class,
            pure=pure,
            far=far,
            lost=lost,
            score=score,
            max_recall=max_recall,
            partner_id_results=partner_id_results,
            clear_status=clear_status,
        )
