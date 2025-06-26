from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    from arcaea_offline_ocr.types import Mat


class OcrTextProvider(ABC):
    @abstractmethod
    def result_raw(self, img: Mat, /, *args, **kwargs) -> Any: ...
    @abstractmethod
    def result(self, img: Mat, /, *args, **kwargs) -> str | None: ...


class ImageCategory(IntEnum):
    JACKET = 0
    PARTNER_ICON = 1


@dataclass(kw_only=True)
class ImageIdProviderResult:
    image_id: str
    category: ImageCategory
    confidence: float


class ImageIdProvider(ABC):
    @abstractmethod
    def result(
        self,
        img: Mat,
        category: ImageCategory,
        /,
        *args,
        **kwargs,
    ) -> ImageIdProviderResult: ...

    @abstractmethod
    def results(
        self,
        img: Mat,
        category: ImageCategory,
        /,
        *args,
        **kwargs,
    ) -> Sequence[ImageIdProviderResult]: ...
