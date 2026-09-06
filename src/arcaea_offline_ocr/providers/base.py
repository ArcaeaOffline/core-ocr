from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

    from cv2.typing import MatLike


class OcrTextProvider(ABC):
    @abstractmethod
    def result_raw(self, img: MatLike, /, *args: Any, **kwargs: Any) -> Any: ...
    @abstractmethod
    def result(self, img: MatLike, /, *args: Any, **kwargs: Any) -> str | None: ...


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
        img: MatLike,
        category: ImageCategory,
        /,
        *args: Any,
        **kwargs: Any,
    ) -> ImageIdProviderResult: ...

    @abstractmethod
    def results(
        self,
        img: MatLike,
        category: ImageCategory,
        /,
        *args: Any,
        **kwargs: Any,
    ) -> Sequence[ImageIdProviderResult]: ...
