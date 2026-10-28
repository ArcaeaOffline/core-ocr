"""Snapshot tests pinning the layout data files and transforms.

Edit the expectations together with deliberate geometry changes;
accidental data or transform edits fail here instead of silently
shifting roi windows.
"""

import pytest

from arcaea_offline_ocr.layouts import load_card_layout
from arcaea_offline_ocr.scenarios.b30.chieri.v4.rois import (
    ChieriBotV4ComponentRois,
)
from arcaea_offline_ocr.scenarios.device.rois.auto import (
    DeviceRoisAutoT1,
    DeviceRoisAutoT2,
)

T1_WIDE = (
    DeviceRoisAutoT1,
    1920,
    1080,
    {
        "pure": [967.5, 780.0, 114.0, 39.0],
        "far": [967.5, 837.0, 114.0, 39.0],
        "lost": [967.5, 891.0, 114.0, 39.0],
        "score": [750.0, 435.0, 420.0, 67.5],
        "rating_class": [45.0, 345.0, 397.5, 52.5],
        "max_recall": [262.5, 292.5, 225.0, 52.5],
        "jacket": [45.0, 400.5, 562.5, 562.5],
        "clear_status": [547.5, 292.5, 330.0, 90.0],
        "partner_icon": [892.5, 0, 135.0, 112.5],
    },
)

T1_NARROW = (
    DeviceRoisAutoT1,
    1440,
    1080,
    {
        "pure": [725.625, 720.0, 85.5, 29.25],
        "far": [725.625, 762.75, 85.5, 29.25],
        "lost": [725.625, 803.25, 85.5, 29.25],
        "score": [562.5, 461.25, 315.0, 50.625],
        "rating_class": [33.75, 393.75, 298.125, 39.375],
        "max_recall": [196.875, 354.375, 168.75, 39.375],
        "jacket": [33.75, 435.375, 421.875, 421.875],
        "clear_status": [410.625, 354.375, 247.5, 67.5],
        "partner_icon": [669.375, 0, 101.25, 84.375],
    },
)

T2_WIDE = (
    DeviceRoisAutoT2,
    2340,
    1080,
    {
        "pure": [1180.0, 790.0, 100.0, 24.0],
        "far": [1180.0, 844.0, 100.0, 24.0],
        "lost": [1180.0, 903.0, 100.0, 24.0],
        "score": [960.0, 435.0, 420.0, 70.0],
        "rating_class": [205.0, 285.0, 350.0, 110.0],
        "max_recall": [545.0, 340.0, 150.0, 50.0],
        "jacket": [255.0, 400.0, 565.0, 565.0],
        "clear_status": [757.5, 290.0, 330.0, 90.0],
        "partner_icon": [1102.5, 0, 135.0, 110.0],
    },
)

T2_NARROW = (
    DeviceRoisAutoT2,
    1440,
    1080,
    {
        "pure": [727.5, 727.5, 75.0, 18.0],
        "far": [727.5, 768.0, 75.0, 18.0],
        "lost": [727.5, 812.25, 75.0, 18.0],
        "score": [562.5, 461.25, 315.0, 52.5],
        "rating_class": [0, 348.75, 262.5, 82.5],
        "max_recall": [251.25, 390.0, 112.5, 37.5],
        "jacket": [33.75, 435.0, 423.75, 423.75],
        "clear_status": [410.625, 352.5, 247.5, 67.5],
        "partner_icon": [669.375, 0, 101.25, 82.5],
    },
)

DEVICE_CASES = [T1_WIDE, T1_NARROW, T2_WIDE, T2_NARROW]


@pytest.mark.parametrize(("cls", "w", "h", "expected"), DEVICE_CASES)
def test_device_rois_snapshot(cls, w, h, expected):
    rois = cls(w, h)
    for field, rect in expected.items():
        actual = getattr(rois, field)
        assert list(actual) == pytest.approx(list(rect), abs=1e-9)
        assert actual.rounded() == tuple(round(v) for v in rect)


CHIERI_CARDS_FACTOR_1 = [
    [107.0, 823.0, 502.0, 240.0],
    [649.0, 823.0, 502.0, 240.0],
    [1191.0, 823.0, 502.0, 240.0],
    [107.0, 1137.0, 502.0, 240.0],
    [649.0, 1137.0, 502.0, 240.0],
    [1191.0, 1137.0, 502.0, 240.0],
    [107.0, 1451.0, 502.0, 240.0],
    [649.0, 1451.0, 502.0, 240.0],
    [1191.0, 1451.0, 502.0, 240.0],
    [107.0, 1765.0, 502.0, 240.0],
    [649.0, 1765.0, 502.0, 240.0],
    [1191.0, 1765.0, 502.0, 240.0],
    [107.0, 2079.0, 502.0, 240.0],
    [649.0, 2079.0, 502.0, 240.0],
    [1191.0, 2079.0, 502.0, 240.0],
    [107.0, 2393.0, 502.0, 240.0],
    [649.0, 2393.0, 502.0, 240.0],
    [1191.0, 2393.0, 502.0, 240.0],
    [107.0, 2707.0, 502.0, 240.0],
    [649.0, 2707.0, 502.0, 240.0],
    [1191.0, 2707.0, 502.0, 240.0],
    [107.0, 3021.0, 502.0, 240.0],
    [649.0, 3021.0, 502.0, 240.0],
    [1191.0, 3021.0, 502.0, 240.0],
    [107.0, 3335.0, 502.0, 240.0],
    [649.0, 3335.0, 502.0, 240.0],
    [1191.0, 3335.0, 502.0, 240.0],
    [107.0, 3649.0, 502.0, 240.0],
    [649.0, 3649.0, 502.0, 240.0],
    [1191.0, 3649.0, 502.0, 240.0],
    [107.0, 4010.0, 502.0, 240.0],
    [649.0, 4010.0, 502.0, 240.0],
    [1191.0, 4010.0, 502.0, 240.0],
]

CHIERI_CARDS_FACTOR_HALF = [
    [53.5, 411.5, 251.0, 120.0],
    [324.5, 411.5, 251.0, 120.0],
    [595.5, 411.5, 251.0, 120.0],
    [53.5, 568.5, 251.0, 120.0],
    [324.5, 568.5, 251.0, 120.0],
    [595.5, 568.5, 251.0, 120.0],
    [53.5, 725.5, 251.0, 120.0],
    [324.5, 725.5, 251.0, 120.0],
    [595.5, 725.5, 251.0, 120.0],
    [53.5, 882.5, 251.0, 120.0],
    [324.5, 882.5, 251.0, 120.0],
    [595.5, 882.5, 251.0, 120.0],
    [53.5, 1039.5, 251.0, 120.0],
    [324.5, 1039.5, 251.0, 120.0],
    [595.5, 1039.5, 251.0, 120.0],
    [53.5, 1196.5, 251.0, 120.0],
    [324.5, 1196.5, 251.0, 120.0],
    [595.5, 1196.5, 251.0, 120.0],
    [53.5, 1353.5, 251.0, 120.0],
    [324.5, 1353.5, 251.0, 120.0],
    [595.5, 1353.5, 251.0, 120.0],
    [53.5, 1510.5, 251.0, 120.0],
    [324.5, 1510.5, 251.0, 120.0],
    [595.5, 1510.5, 251.0, 120.0],
    [53.5, 1667.5, 251.0, 120.0],
    [324.5, 1667.5, 251.0, 120.0],
    [595.5, 1667.5, 251.0, 120.0],
    [53.5, 1824.5, 251.0, 120.0],
    [324.5, 1824.5, 251.0, 120.0],
    [595.5, 1824.5, 251.0, 120.0],
    [53.5, 2005.0, 251.0, 120.0],
    [324.5, 2005.0, 251.0, 120.0],
    [595.5, 2005.0, 251.0, 120.0],
]


def test_chieri_card_layout_snapshot():
    layout = load_card_layout("chieri_v4")
    assert layout.component_count == 33


def test_chieri_cards_snapshot():
    layout = load_card_layout("chieri_v4")
    for i, expected in enumerate(CHIERI_CARDS_FACTOR_1):
        assert list(layout.card_rect(i, 1.0)) == pytest.approx(expected, abs=1e-9)
    for i, expected in enumerate(CHIERI_CARDS_FACTOR_HALF):
        assert list(layout.card_rect(i, 0.5)) == pytest.approx(expected, abs=1e-9)


def test_chieri_component_rois_snapshot():
    rois = ChieriBotV4ComponentRois(1.0)
    assert list(rois.rating_class_rect) == pytest.approx([21, 40, 7, 20])
    assert list(rois.jacket_rect) == pytest.approx([263, 0, 239, 239])
    assert list(rois.score_rect) == pytest.approx([30, 60, 270, 55])
    assert list(rois.pfl_rect) == pytest.approx([50, 125, 80, 100])
    assert rois.bg_point == pytest.approx((75, 10))
