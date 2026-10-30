"""Roi layout definitions, loaded from the JSON files packaged here.

Every rect in a layout file is ``[x, y, w, h]`` from the top-left
corner of the file's own coordinate space.

``device/*.json`` files author field windows on the design canvas the
file declares; ``DeviceLayout.screen_rect`` maps a window onto a
screenshot - screens with the canvas aspect or wider scale by height,
narrower ones scale by width and center the content vertically.

``b30/*.json`` files describe bot-rendered images with a fixed pixel
canvas: an ordered list of card origins (list index = component
index) plus card-local field windows and probe points, all scaled
uniformly by image width.
"""

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
    design: tuple[int, int]
    fields: Mapping[str, Rect]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """Parse a ``device/*.json`` file mapping."""
        return cls(
            design=tuple(data["design"]),
            fields={k: tuple(v) for k, v in data["fields"].items()},
        )

    def screen_rect(self, field: str, w: int, h: int) -> XYWHRect:
        x_d, y_d, w_d, h_d = self.fields[field]
        dw, dh = self.design
        f = min(w / dw, h / dh)
        x = max(0.0, w / 2 + (x_d - dw / 2) * f)
        y = y_d * f if (w / h) >= (dw / dh) else h / 2 + (y_d - dh / 2) * f
        return XYWHRect(x, y, w_d * f, h_d * f)


@dataclass(frozen=True)
class CardLayout:
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
