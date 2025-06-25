from abc import abstractmethod

from arcaea_offline_ocr.scenarios.base import OcrScenario, OcrScenarioResult


class DeviceScenarioBase(OcrScenario):
    @abstractmethod
    def result(self) -> OcrScenarioResult: ...
