"""Snapshot tests pinning the layout data files and transforms.

Edit the expectations together with deliberate geometry changes;
accidental data or transform edits fail here instead of silently
shifting roi windows.
"""

import pytest

from arcaea_offline_ocr.layouts import load_builtin_card_layout
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
        "score": [735.0, 433.5, 450.0, 70.5],
        "rating_class": [45.0, 345.0, 270.0, 52.5],
        "max_recall": [262.5, 292.5, 225.0, 52.5],
        "jacket": [45.0, 400.5, 562.5, 562.5],
        "clear_status": [547.5, 292.5, 330.0, 90.0],
        "partner_icon": [892.5, 0.0, 135.0, 112.5],
        "high_score": [900.0, 537.0, 225.0, 45.0],
        "high_score_delta": [1134.0, 540.0, 174.0, 39.0],
        "pure_shiny": [1074.0, 781.5, 112.5, 45.0],
        "early_late": [1071.0, 834.0, 375.0, 48.0],
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
        "score": [551.25, 460.125, 337.5, 52.875],
        "rating_class": [33.75, 393.75, 202.5, 39.375],
        "max_recall": [196.875, 354.375, 168.75, 39.375],
        "jacket": [33.75, 435.375, 421.875, 421.875],
        "clear_status": [410.625, 354.375, 247.5, 67.5],
        "partner_icon": [669.375, 0.0, 101.25, 84.375],
        "high_score": [675.0, 537.75, 168.75, 33.75],
        "high_score_delta": [850.5, 540.0, 130.5, 29.25],
        "pure_shiny": [805.5, 721.125, 84.375, 33.75],
        "early_late": [803.25, 760.5, 281.25, 36.0],
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
        "pure": [1179.0, 787.5, 100.5, 30.0],
        "far": [1179.0, 843.75, 100.5, 30.0],
        "lost": [1179.0, 900.0, 100.5, 30.0],
        "score": [945.0, 433.5, 450.0, 69.0],
        "rating_class": [205.5, 285.0, 349.5, 109.5],
        "max_recall": [544.5, 339.0, 150.0, 49.5],
        "jacket": [255.0, 399.0, 564.0, 564.0],
        "clear_status": [757.5, 289.5, 330.0, 90.0],
        "partner_icon": [1102.5, 0.0, 135.0, 110.0],
        "high_score": [1134.0, 529.5, 195.0, 55.5],
        "high_score_delta": [1350.0, 541.5, 165.0, 36.0],
        "pure_shiny": [1282.5, 787.5, 127.5, 30.0],
        "early_late": [1282.5, 840.75, 375.0, 36.0],
        "pure_label": [1071.0, 778.5, 82.5, 48.0],
        "far_label": [1083.0, 833.25, 60.0, 48.0],
        "lost_label": [1072.5, 888.75, 81.0, 48.0],
    },
)

T2_NARROW = (
    DeviceRoisAutoT2,
    1440,
    1080,
    {
        "pure": [726.75, 725.625, 75.375, 22.5],
        "far": [726.75, 767.8125, 75.375, 22.5],
        "lost": [726.75, 810.0, 75.375, 22.5],
        "score": [551.25, 460.125, 337.5, 51.75],
        "rating_class": [0.0, 348.75, 262.125, 82.125],
        "max_recall": [250.875, 389.25, 112.5, 37.125],
        "jacket": [33.75, 434.25, 423.0, 423.0],
        "clear_status": [410.625, 352.125, 247.5, 67.5],
        "partner_icon": [669.375, 0.0, 101.25, 82.5],
        "high_score": [693.0, 532.125, 146.25, 41.625],
        "high_score_delta": [855.0, 541.125, 123.75, 27.0],
        "pure_shiny": [804.375, 725.625, 95.625, 22.5],
        "early_late": [804.375, 765.5625, 281.25, 27.0],
        "pure_label": [645.75, 718.875, 61.875, 36.0],
        "far_label": [654.75, 759.9375, 45.0, 36.0],
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
    layout = load_builtin_card_layout("chieri_v4")
    assert layout.component_count == 33


def test_chieri_cards_snapshot():
    layout = load_builtin_card_layout("chieri_v4")
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
