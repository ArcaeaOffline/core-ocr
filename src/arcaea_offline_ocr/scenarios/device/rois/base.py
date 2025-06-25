from abc import ABC, abstractmethod

from arcaea_offline_ocr.types import XYWHRect


class DeviceRois(ABC):
    @property
    @abstractmethod
    def pure(self) -> XYWHRect: ...
    @property
    @abstractmethod
    def far(self) -> XYWHRect: ...
    @property
    @abstractmethod
    def lost(self) -> XYWHRect: ...
    @property
    @abstractmethod
    def score(self) -> XYWHRect: ...
    @property
    @abstractmethod
    def rating_class(self) -> XYWHRect: ...
    @property
    @abstractmethod
    def max_recall(self) -> XYWHRect: ...
    @property
    @abstractmethod
    def jacket(self) -> XYWHRect: ...
    @property
    @abstractmethod
    def clear_status(self) -> XYWHRect: ...
    @property
    @abstractmethod
    def partner_icon(self) -> XYWHRect: ...
