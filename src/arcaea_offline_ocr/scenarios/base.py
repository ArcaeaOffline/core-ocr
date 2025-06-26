from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from datetime import datetime

    from arcaea_offline_ocr.providers import ImageIdProviderResult


@dataclass(kw_only=True)
class OcrScenarioResult:
    song_id: str
    rating_class: int
    score: int

    song_id_results: Sequence[ImageIdProviderResult] = field(default_factory=list)
    partner_id_results: Sequence[ImageIdProviderResult] = field(
        default_factory=list,
    )

    pure: int | None = None
    pure_inaccurate: int | None = None
    pure_early: int | None = None
    pure_late: int | None = None
    far: int | None = None
    far_inaccurate: int | None = None
    far_early: int | None = None
    far_late: int | None = None
    lost: int | None = None

    played_at: datetime | None = None
    max_recall: int | None = None
    clear_status: int | None = None
    clear_type: int | None = None
    modifier: int | None = None


class OcrScenario(ABC):  # noqa: B024
    pass
