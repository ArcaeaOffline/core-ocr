from abc import ABC, abstractmethod

from arcaea_offline_ocr.scenarios.base import OcrScenario, OcrScenarioResult


class DeviceScenarioBase(OcrScenario, ABC):
    @abstractmethod
    def result(self) -> OcrScenarioResult: ...
