from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from ..types import Mat


class OcrTextProvider(ABC):
    @abstractmethod
    def result_raw(self, img: "Mat", /, *args, **kwargs) -> Any: ...
    @abstractmethod
    def result(self, img: "Mat", /, *args, **kwargs) -> Optional[str]: ...
