"""Roi layout definitions loaded from packaged JSON files."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from importlib import resources
from typing import TYPE_CHECKING, Any

from arcaea_offline_ocr.types import XYWHRect

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Self

Rect = tuple[float, float, float, float]
Point = tuple[float, float]


def _load_json(group: str, name: str) -> dict[str, Any]:
    path = resources.files("arcaea_offline_ocr.layouts").joinpath(group, f"{name}.json")
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class DeviceLayout:
    """Field windows authored in design-canvas coordinates.

    Rects are ``[x, y, w, h]`` measured from the canvas top-left. Map
    them onto a screenshot with ``screen_rect``.
    """

    design: tuple[int, int]
    fields: Mapping[str, Rect]
    tablet_max_aspect: float | None = None
    tablet_height_ratio: float = 4 / 3

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """Parse a ``device/*.json`` file mapping."""
        return cls(
            design=tuple(data["design"]),
            fields={k: tuple(v) for k, v in data["fields"].items()},
            tablet_max_aspect=data.get("tablet_max_aspect"),
            tablet_height_ratio=data.get("tablet_height_ratio") or 4 / 3,
        )

    def screen_rect(self, field: str, w: int, h: int) -> XYWHRect:
        """Map a field window onto a ``w`` x ``h`` screenshot.

        Screens at the design aspect or wider scale by height and anchor
        the content to the screen top; narrower ones scale by width and
        center the content vertically. A layout declaring
        ``tablet_max_aspect`` renders narrower screenshots below that
        aspect inside a canvas ``tablet_height_ratio`` times the design
        height instead: the design panel stays centered inside the
        taller stage.
        """
        x_d, y_d, w_d, h_d = self.fields[field]
        dw, dh = self.design
        if self.tablet_max_aspect is not None and (w / h) < self.tablet_max_aspect:
            f = min(w / dw, h / (dh * self.tablet_height_ratio))
        else:
            f = min(w / dw, h / dh)
        x = max(0.0, w / 2 + (x_d - dw / 2) * f)
        y = h / 2 + (y_d - dh / 2) * f
        return XYWHRect(x, y, w_d * f, h_d * f)


@dataclass(frozen=True)
class CardLayout:
    """Layout of a bot-rendered b30 image on a fixed pixel canvas: card
    origins ordered by component index, card-local field windows and
    probe points, all scaled uniformly by image width.
    """

    canvas: tuple[int, int]
    card_size: tuple[float, float]
    cards: Sequence[Point]
    fields: Mapping[str, Rect]
    points: Mapping[str, Point]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """Parse a ``b30/*.json`` file mapping."""
        return cls(
            canvas=tuple(data["canvas"]),
            card_size=tuple(data["card_size"]),
            cards=tuple(tuple(p) for p in data["cards"]),
            fields={k: tuple(v) for k, v in data["fields"].items()},
            points={k: tuple(v) for k, v in data.get("points", {}).items()},
        )

    @property
    def component_count(self) -> int:
        return len(self.cards)

    def factor(self, w: int) -> float:
        return w / self.canvas[0]

    def card_rect(self, index: int, factor: float) -> XYWHRect:
        x, y = self.cards[index]
        w, h = self.card_size
        return XYWHRect(x, y, w, h) * factor

    def field_rect(self, field: str, factor: float) -> XYWHRect:
        return XYWHRect(*self.fields[field]) * factor

    def point(self, name: str, factor: float) -> Point:
        x, y = self.points[name]
        return (x * factor, y * factor)


@cache
def load_builtin_device_layout(name: str) -> DeviceLayout:
    return DeviceLayout.from_dict(_load_json("device", name))


@cache
def load_builtin_card_layout(name: str) -> CardLayout:
    return CardLayout.from_dict(_load_json("b30", name))


__all__ = ["CardLayout", "DeviceLayout"]
