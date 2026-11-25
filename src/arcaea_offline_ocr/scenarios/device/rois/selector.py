from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from typing_extensions import override

from arcaea_offline_ocr.scenarios.device.masker.auto import (
    DeviceRoisMaskerAuto,
    DeviceRoisMaskerAutoT1,
    DeviceRoisMaskerAutoT2,
)
from arcaea_offline_ocr.scenarios.device.masker.rgb_gray import RgbGrayMasker

from .auto import DeviceRoisAutoT1, DeviceRoisAutoT2

if TYPE_CHECKING:
    from collections.abc import Callable

    from cv2.typing import MatLike

    from arcaea_offline_ocr.scenarios.device.masker.base import DeviceRoisMasker

    from .base import DeviceRois

logger = logging.getLogger(__name__)


class DeviceRoisAutoSelectorResult(Enum):
    T1 = "T1"
    T2 = "T2"
    UNKNOWN = "UNKNOWN"


def crop_rounded(img_bgr: MatLike, rect: tuple[float, float, float, float]) -> MatLike:
    x, y, w, h = (round(v) for v in rect)
    return img_bgr[y : y + h, x : x + w]


@dataclass
class LabelMatch:
    """
    One interest point check: the non-zero pixel ratio of its masked label ROI
    against the threshold.
    """

    name: str
    ratio: float
    threshold: float

    @property
    def matched(self) -> bool:
        return self.ratio >= self.threshold


@dataclass
class DetectionReport:
    labels: list[LabelMatch]

    @property
    def confidence(self) -> float:
        """Fraction of labels that matched, in the ``[0, 1]`` range."""
        return sum(label.matched for label in self.labels) / len(self.labels)


class DeviceRoisAutoSelectorDetector(ABC):
    result: ClassVar[DeviceRoisAutoSelectorResult]

    @abstractmethod
    def analyze(self, img_bgr: MatLike) -> DetectionReport:
        """
        Analyze the img_bgr and report each interest point's mask ratio against
        its threshold.

        For example, if the detector has 3 interest points and 2 of them match,
        the report's confidence would be ``0.66``.
        """


@dataclass
class _Interest:
    name: str
    rect: tuple[float, float, float, float]
    masker: Callable[[MatLike], MatLike]
    non_zero_ratio_threshold: float


def _analyze_labels(img_bgr: MatLike, interests: list[_Interest]) -> list[LabelMatch]:
    labels: list[LabelMatch] = []
    for interest in interests:
        label_bgr = crop_rounded(img_bgr, interest.rect)
        label_masked = interest.masker(label_bgr)
        labels.append(
            LabelMatch(
                interest.name,
                np.count_nonzero(label_masked) / label_masked.size,
                interest.non_zero_ratio_threshold,
            ),
        )
    return labels


@dataclass
class _StripMatch:
    purple: LabelMatch
    gray: LabelMatch


class _StripProbe:
    """
    The top-left background strip check that gates label detection: purple is
    the regular Arcaea skin, gray (BGR 10..40) is the Arknights collab skin,
    and any other background cannot be an Arcaea result screenshot. The strip
    anchors to T2 geometry and holds for T1 screenshots as well.
    """

    HEIGHT_OFFSET: ClassVar[int] = -510
    WIDTH_OFFSET: ClassVar[int] = -250
    HEIGHT: ClassVar[int] = 165

    PURPLE_HSV_LOWER: ClassVar[np.ndarray] = np.array([110, 45, 75], np.uint8)
    PURPLE_HSV_UPPER: ClassVar[np.ndarray] = np.array([140, 150, 175], np.uint8)
    PURPLE_NON_ZERO_RATIO_THRESHOLD: ClassVar[float] = 0.65

    GRAY_BGR_LOWER: ClassVar[np.ndarray] = np.array([10] * 3, np.uint8)
    GRAY_BGR_UPPER: ClassVar[np.ndarray] = np.array([40] * 3, np.uint8)
    GRAY_NON_ZERO_RATIO_THRESHOLD: ClassVar[float] = 0.5

    @classmethod
    def analyze(cls, img_bgr: MatLike) -> _StripMatch:
        rois = DeviceRoisAutoT2(img_bgr.shape[1], img_bgr.shape[0])

        y = round(rois.layout_area_h_mid + cls.HEIGHT_OFFSET * rois.factor)
        w = round(rois.w_mid + cls.WIDTH_OFFSET * rois.factor)
        h = round(cls.HEIGHT * rois.factor)

        strip_bgr = img_bgr[y : y + h, 0:w]
        purple_mask = DeviceRoisMaskerAuto.mask_bgr_in_hsv(
            strip_bgr,
            cls.PURPLE_HSV_LOWER,
            cls.PURPLE_HSV_UPPER,
        )
        gray_mask = RgbGrayMasker.gray(
            strip_bgr,
            cls.GRAY_BGR_LOWER,
            cls.GRAY_BGR_UPPER,
        )
        return _StripMatch(
            LabelMatch(
                "purple",
                np.count_nonzero(purple_mask) / purple_mask.size,
                cls.PURPLE_NON_ZERO_RATIO_THRESHOLD,
            ),
            LabelMatch(
                "gray",
                np.count_nonzero(gray_mask) / gray_mask.size,
                cls.GRAY_NON_ZERO_RATIO_THRESHOLD,
            ),
        )


class DeviceRoisAutoSelectorDetectorT1(DeviceRoisAutoSelectorDetector):
    PURE_LABEL_HSV_LOWER: ClassVar[np.ndarray] = np.array([80, 60, 125], np.uint8)
    PURE_LABEL_HSV_UPPER: ClassVar[np.ndarray] = np.array([110, 200, 225], np.uint8)
    FAR_LOST_LABEL_GRAY_BGR_LOWER: ClassVar[np.ndarray] = np.array([120] * 3, np.uint8)
    FAR_LOST_LABEL_GRAY_BGR_UPPER: ClassVar[np.ndarray] = np.array([160] * 3, np.uint8)

    PURE_LABEL_NON_ZERO_RATIO_THRESHOLD: ClassVar[float] = 0.08
    FAR_LABEL_NON_ZERO_RATIO_THRESHOLD: ClassVar[float] = 0.06
    LOST_LABEL_NON_ZERO_RATIO_THRESHOLD: ClassVar[float] = 0.06

    result: ClassVar[DeviceRoisAutoSelectorResult] = DeviceRoisAutoSelectorResult.T1

    @classmethod
    def _mask_pure_label(cls, label_bgr: MatLike) -> MatLike:
        return DeviceRoisMaskerAuto.mask_bgr_in_hsv(
            label_bgr,
            cls.PURE_LABEL_HSV_LOWER,
            cls.PURE_LABEL_HSV_UPPER,
        )

    @classmethod
    def _mask_far_lost_label(cls, label_bgr: MatLike) -> MatLike:
        return RgbGrayMasker.gray(
            label_bgr,
            cls.FAR_LOST_LABEL_GRAY_BGR_LOWER,
            cls.FAR_LOST_LABEL_GRAY_BGR_UPPER,
        )

    @override
    def analyze(self, img_bgr: MatLike) -> DetectionReport:
        rois = DeviceRoisAutoT1(img_bgr.shape[1], img_bgr.shape[0])

        labels = _analyze_labels(
            img_bgr,
            [
                _Interest(
                    "pure",
                    rois.pure_label,
                    self._mask_pure_label,
                    self.PURE_LABEL_NON_ZERO_RATIO_THRESHOLD,
                ),
                _Interest(
                    "far",
                    rois.far_label,
                    self._mask_far_lost_label,
                    self.FAR_LABEL_NON_ZERO_RATIO_THRESHOLD,
                ),
                _Interest(
                    "lost",
                    rois.lost_label,
                    self._mask_far_lost_label,
                    self.LOST_LABEL_NON_ZERO_RATIO_THRESHOLD,
                ),
            ],
        )
        return DetectionReport(labels)


class DeviceRoisAutoSelectorDetectorT2(DeviceRoisAutoSelectorDetector):
    PURE_LABEL_HSV_LOWER: ClassVar[np.ndarray] = np.array([110, 25, 90], np.uint8)
    PURE_LABEL_HSV_UPPER: ClassVar[np.ndarray] = np.array([160, 150, 230], np.uint8)
    FAR_LABEL_HSV_LOWER: ClassVar[np.ndarray] = np.array([5, 25, 120], np.uint8)
    FAR_LABEL_HSV_UPPER: ClassVar[np.ndarray] = np.array([20, 100, 240], np.uint8)
    LOST_LABEL_HSV_LOWER: ClassVar[np.ndarray] = np.array([160, 5, 190], np.uint8)
    LOST_LABEL_HSV_UPPER: ClassVar[np.ndarray] = np.array([179, 60, 255], np.uint8)

    PFL_LABEL_NON_ZERO_RATIO_THRESHOLD: ClassVar[float] = 0.3

    result: ClassVar[DeviceRoisAutoSelectorResult] = DeviceRoisAutoSelectorResult.T2

    @classmethod
    def _hsv_interest(
        cls,
        name: str,
        rect: tuple[float, float, float, float],
        hsv_lower: np.ndarray,
        hsv_upper: np.ndarray,
    ) -> _Interest:
        return _Interest(
            name,
            rect,
            lambda label_bgr: DeviceRoisMaskerAuto.mask_bgr_in_hsv(
                label_bgr,
                hsv_lower,
                hsv_upper,
            ),
            cls.PFL_LABEL_NON_ZERO_RATIO_THRESHOLD,
        )

    @override
    def analyze(self, img_bgr: MatLike) -> DetectionReport:
        rois = DeviceRoisAutoT2(img_bgr.shape[1], img_bgr.shape[0])

        labels = _analyze_labels(
            img_bgr,
            [
                self._hsv_interest(
                    "pure",
                    rois.pure_label,
                    self.PURE_LABEL_HSV_LOWER,
                    self.PURE_LABEL_HSV_UPPER,
                ),
                self._hsv_interest(
                    "far",
                    rois.far_label,
                    self.FAR_LABEL_HSV_LOWER,
                    self.FAR_LABEL_HSV_UPPER,
                ),
                self._hsv_interest(
                    "lost",
                    rois.lost_label,
                    self.LOST_LABEL_HSV_LOWER,
                    self.LOST_LABEL_HSV_UPPER,
                ),
            ],
        )
        return DetectionReport(labels)


class DeviceRoisAutoSelectorDetectorT2ArknightsSkin(DeviceRoisAutoSelectorDetector):
    PFL_LABEL_HSV_LOWER: ClassVar[np.ndarray] = np.array([0, 5, 70], np.uint8)
    PFL_LABEL_HSV_UPPER: ClassVar[np.ndarray] = np.array([179, 90, 145], np.uint8)

    PFL_LABEL_NON_ZERO_RATIO_THRESHOLD: ClassVar[float] = 0.7

    result: ClassVar[DeviceRoisAutoSelectorResult] = DeviceRoisAutoSelectorResult.T2

    @classmethod
    def _mask_pfl_label(cls, label_bgr: MatLike) -> MatLike:
        return DeviceRoisMaskerAuto.mask_bgr_in_hsv(
            label_bgr,
            cls.PFL_LABEL_HSV_LOWER,
            cls.PFL_LABEL_HSV_UPPER,
        )

    @override
    def analyze(self, img_bgr: MatLike) -> DetectionReport:
        rois = DeviceRoisAutoT2(img_bgr.shape[1], img_bgr.shape[0])

        labels = _analyze_labels(
            img_bgr,
            [
                _Interest(
                    "pure",
                    rois.pure_label,
                    self._mask_pfl_label,
                    self.PFL_LABEL_NON_ZERO_RATIO_THRESHOLD,
                ),
                _Interest(
                    "far",
                    rois.far_label,
                    self._mask_pfl_label,
                    self.PFL_LABEL_NON_ZERO_RATIO_THRESHOLD,
                ),
                _Interest(
                    "lost",
                    rois.lost_label,
                    self._mask_pfl_label,
                    self.PFL_LABEL_NON_ZERO_RATIO_THRESHOLD,
                ),
            ],
        )
        return DetectionReport(labels)


def _format_label(label: LabelMatch) -> str:
    matched_text = "Y" if label.matched else "n"
    return f"{label.name}[{matched_text}]: {label.ratio:.3f}/{label.threshold:.3f}"


def _log_select_results(
    strip: _StripMatch,
    results: dict[DeviceRoisAutoSelectorDetector, DetectionReport],
) -> None:
    if not logger.isEnabledFor(logging.DEBUG):
        return

    lines: list[str] = []
    for detector, report in results.items():
        detector_name = type(detector).__name__
        result_name = detector.result.name
        labels = ", ".join(_format_label(label) for label in report.labels)
        text = f"\t{detector_name} ({result_name}, {report.confidence:.3f}) - {labels}"
        lines.append(text)
    results_separated = "\n".join(lines) or "\t(none)"
    logger.debug(
        "Strip - %s, %s\nDetector results:\n%s",
        _format_label(strip.purple),
        _format_label(strip.gray),
        results_separated,
    )


class DeviceRoisAutoSelector:
    @classmethod
    def select(
        cls,
        img_bgr: MatLike,
        fallback: DeviceRoisAutoSelectorResult | None = None,
    ) -> DeviceRoisAutoSelectorResult:
        """
        Classify an Arcaea result screenshot into a rois type by its top-left
        background strip.

        ``fallback`` is selected when no skin matches or every detector reports
        zero confidence.
        """
        strip = _StripProbe.analyze(img_bgr)

        detectors: list[DeviceRoisAutoSelectorDetector]
        if strip.purple.matched:
            detectors = [
                DeviceRoisAutoSelectorDetectorT1(),
                DeviceRoisAutoSelectorDetectorT2(),
            ]
        elif strip.gray.matched:
            detectors = [DeviceRoisAutoSelectorDetectorT2ArknightsSkin()]
        else:
            detectors = []

        results = {detector: detector.analyze(img_bgr) for detector in detectors}
        _log_select_results(strip, results)

        if all(report.confidence == 0.0 for report in results.values()):
            if fallback is not None:
                return fallback
            return DeviceRoisAutoSelectorResult.UNKNOWN
        return max(results.items(), key=lambda item: item[1].confidence)[0].result

    @staticmethod
    def create_rois_and_masker(
        result: DeviceRoisAutoSelectorResult,
        w: int,
        h: int,
    ) -> tuple[DeviceRois, DeviceRoisMasker]:
        """Create the rois and masker pair matching the selector result."""
        if result is DeviceRoisAutoSelectorResult.T1:
            return DeviceRoisAutoT1(w, h), DeviceRoisMaskerAutoT1()
        if result is DeviceRoisAutoSelectorResult.T2:
            return DeviceRoisAutoT2(w, h), DeviceRoisMaskerAutoT2()
        msg = f"no rois/masker pair for {result}"
        raise ValueError(msg)
