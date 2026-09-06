from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from arcaea_offline_ocr.scenarios.base import OcrScenario, OcrScenarioResult

if TYPE_CHECKING:
    from cv2.typing import MatLike


class Best30Scenario(OcrScenario, ABC):
    @abstractmethod
    def components(self, img: MatLike, /) -> list[MatLike]: ...

    @abstractmethod
    def result(
        self,
        component_img: MatLike,
        /,
        *args: Any,
        **kwargs: Any,
    ) -> OcrScenarioResult: ...

    @abstractmethod
    def results(
        self, img: MatLike, /, *args: Any, **kwargs: Any
    ) -> list[OcrScenarioResult]:
        """
        Commonly a shorthand for `[self.result(comp) for comp in self.components(img)]`
        """
        ...
