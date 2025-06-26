from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from arcaea_offline_ocr.scenarios.base import OcrScenario, OcrScenarioResult

if TYPE_CHECKING:
    from arcaea_offline_ocr.types import Mat


class Best30Scenario(OcrScenario):
    @abstractmethod
    def components(self, img: Mat, /) -> list[Mat]: ...

    @abstractmethod
    def result(self, component_img: Mat, /, *args, **kwargs) -> OcrScenarioResult: ...

    @abstractmethod
    def results(self, img: Mat, /, *args, **kwargs) -> list[OcrScenarioResult]:
        """
        Commonly a shorthand for `[self.result(comp) for comp in self.components(img)]`
        """
        ...
