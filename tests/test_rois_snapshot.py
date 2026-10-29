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
        "high_score": [864.0, 529.5, 279.0, 60.0],
        "high_score_delta": [1155.0, 529.5, 177.0, 60.0],
        "pure_shiny": [1074.0, 781.5, 93.0, 45.0],
        "early_late": [1071.0, 834.0, 412.5, 48.0],
        "pure_label": [858.0, 774.855, 91.5, 55.5],
        "far_label": [883.5, 829.1999999999999, 66.0, 55.5],
        "lost_label": [861.0, 884.355, 88.5, 55.5],
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
        "high_score": [648.0, 532.125, 209.25, 45.0],
        "high_score_delta": [866.25, 532.125, 132.75, 45.0],
        "pure_shiny": [805.5, 721.125, 69.75, 33.75],
        "early_late": [803.25, 760.5, 309.375, 36.0],
        "pure_label": [643.5, 716.1412500000001, 68.625, 41.625],
        "far_label": [662.625, 756.9, 49.5, 41.625],
        "lost_label": [645.75, 798.2662500000001, 66.375, 41.625],
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
        "high_score": [1150.0, 530.0, 190.0, 55.0],
        "high_score_delta": [1360.0, 530.0, 160.0, 55.0],
        "pure_shiny": [1283.0, 773.0, 108.0, 56.0],
        "early_late": [1282.0, 828.0, 275.0, 62.0],
        "pure_label": [1071.0, 778.5, 82.5, 48.0],
        "far_label": [1083.0, 833.205, 60.0, 48.0],
        "lost_label": [1072.5, 888.75, 81.0, 48.0],
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
        "high_score": [705.0, 532.5, 142.5, 41.25],
        "high_score_delta": [862.5, 532.5, 120.0, 41.25],
        "pure_shiny": [804.75, 714.75, 81.0, 42.0],
        "early_late": [804.0, 756.0, 206.25, 46.5],
        "pure_label": [645.75, 718.875, 61.875, 36.0],
        "far_label": [654.75, 759.9037500000001, 45.0, 36.0],
        "lost_label": [646.875, 801.5625, 60.75, 36.0],
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
    assert list(rois.pure_rect) == pytest.approx([64, 134, 62, 20])
    assert list(rois.far_rect) == pytest.approx([64, 167, 62, 20])
    assert list(rois.lost_rect) == pytest.approx([64, 201, 62, 20])
