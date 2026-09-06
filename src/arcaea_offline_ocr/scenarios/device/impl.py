import cv2
import numpy as np
from cv2.typing import MatLike

from arcaea_offline_ocr.providers import (
    ImageCategory,
    ImageIdProvider,
    OcrCrnnTextProvider,
)
from arcaea_offline_ocr.scenarios.base import OcrScenarioResult

from .base import DeviceScenarioBase
from .extractor import DeviceRoisExtractor
from .masker import DeviceRoisMasker


class DeviceScenario(DeviceScenarioBase):
    extractor: DeviceRoisExtractor
    masker: DeviceRoisMasker
    crnn_provider: OcrCrnnTextProvider
    image_id_provider: ImageIdProvider

    def __init__(
        self,
        extractor: DeviceRoisExtractor,
        masker: DeviceRoisMasker,
        crnn_provider: OcrCrnnTextProvider,
        image_id_provider: ImageIdProvider,
    ):
        self.extractor = extractor
        self.masker = masker
        self.crnn_provider = crnn_provider
        self.image_id_provider = image_id_provider

    def pure(self):
        ocr_result = self.crnn_provider.result(self.extractor.pure)
        return int(ocr_result) if ocr_result else 0

    def far(self):
        ocr_result = self.crnn_provider.result(self.extractor.far)
        return int(ocr_result) if ocr_result else 0

    def lost(self):
        ocr_result = self.crnn_provider.result(self.extractor.lost)
        return int(ocr_result) if ocr_result else 0

    def score(self):
        ocr_result = self.crnn_provider.result(self.extractor.score)
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
        ocr_result = self.crnn_provider.result(self.extractor.max_recall)
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
    def preprocess_char_icon(img_gray: MatLike):
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
