from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

    import cv2


class OcrTextProvider(ABC):
    @abstractmethod
    def result_raw(self, img: cv2.typing.MatLike, /, *args, **kwargs) -> Any: ...
    @abstractmethod
    def result(self, img: cv2.typing.MatLike, /, *args, **kwargs) -> str | None: ...


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
        img: cv2.typing.MatLike,
        category: ImageCategory,
        /,
        *args,
        **kwargs,
    ) -> ImageIdProviderResult: ...

    @abstractmethod
    def results(
        self,
        img: cv2.typing.MatLike,
        category: ImageCategory,
        /,
        *args,
        **kwargs,
    ) -> Sequence[ImageIdProviderResult]: ...
