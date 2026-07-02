from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from arcaea_offline_ocr.scenarios.base import OcrScenario, OcrScenarioResult

if TYPE_CHECKING:
    from cv2.typing import MatLike


class Best30Scenario(OcrScenario):
    @abstractmethod
    def components(self, img: MatLike, /) -> list[MatLike]: ...

    @abstractmethod
    def result(
        self, component_img: MatLike, /, *args, **kwargs
    ) -> OcrScenarioResult: ...

    @abstractmethod
    def results(self, img: MatLike, /, *args, **kwargs) -> list[OcrScenarioResult]:
        """
        Commonly a shorthand for `[self.result(comp) for comp in self.components(img)]`
        """
        ...
