from base64 import b64decode
from time import sleep
from typing import Dict, List, Literal, Tuple, TypedDict

from cv2 import (
    CHAIN_APPROX_SIMPLE,
    COLOR_BGR2GRAY,
    COLOR_GRAY2BGR,
    FONT_HERSHEY_SIMPLEX,
    IMREAD_GRAYSCALE,
    RETR_EXTERNAL,
    THRESH_BINARY_INV,
    TM_CCOEFF_NORMED,
    Mat,
    boundingRect,
    cvtColor,
    destroyAllWindows,
    findContours,
    imdecode,
    imread,
    imshow,
    matchTemplate,
    minMaxLoc,
    putText,
    rectangle,
    threshold,
    waitKey,
)
from imutils import contours, grab_contours
from numpy import frombuffer as np_frombuffer
from numpy import uint8

from ._builtin_templates import GeoSansLight_Italic, GeoSansLight_Regular


def load_digit_template(filename: str) -> Dict[int, Mat]:
    """
    Arguments:
        filename -- An image with white background and black "0 1 2 3 4 5 6 7 8 9" text.

    Returns:
        dict[int, cv2.Mat]
    """
    # https://pyimagesearch.com/2017/07/17/credit-card-ocr-with-opencv-and-python/
    ref = imread(filename)
    ref = cvtColor(ref, COLOR_BGR2GRAY)
    ref = threshold(ref, 10, 255, THRESH_BINARY_INV)[1]
    refCnts = findContours(ref.copy(), RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)
    refCnts = grab_contours(refCnts)
    refCnts = contours.sort_contours(refCnts, method="left-to-right")[0]
    digits = {}
    for i, cnt in enumerate(refCnts):
        (x, y, w, h) = boundingRect(cnt)
        roi = ref[y : y + h, x : x + w]
        digits[i] = roi
    return digits


def load_builtin_digit_template(
    name: Literal["GeoSansLight-Regular", "GeoSansLight-Italic"]
):
    name_builtin_template_b64_map = {
        "GeoSansLight-Regular": GeoSansLight_Regular,
        "GeoSansLight-Italic": GeoSansLight_Italic,
    }
    template_b64 = name_builtin_template_b64_map[name]
    return {
        int(key): imdecode(np_frombuffer(b64decode(b64str), uint8), IMREAD_GRAYSCALE)
        for key, b64str in template_b64.items()
    }


class MatchTemplateMultipleResult(TypedDict):
    max_val: float
    xywh: Tuple[int, int, int, int]


def matchTemplateMultiple(
    src: Mat, template: Mat, threshold: float = 0.1
) -> List[MatchTemplateMultipleResult]:
    """
    Returns:
        A list of tuple[x, y, w, h] representing the matched rectangle
    """
    template_result = matchTemplate(src, template, TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = minMaxLoc(template_result)
    template_h, template_w = template.shape[:2]
    results = []

    # debug
    # imshow("templ", template)
    # waitKey(750)
    # destroyAllWindows()

    # https://stackoverflow.com/a/66848923/16484891
    # CC BY-SA 4.0
    prev_min_val, prev_max_val, prev_min_loc, prev_max_loc = None, None, None, None
    while max_val > threshold:
        min_val, max_val, min_loc, max_loc = minMaxLoc(template_result)

        # Prevent infinite loop. If those 4 values are the same as previous ones, break the loop.
        if (
            prev_min_val == min_val
            and prev_max_val == max_val
            and prev_min_loc == min_loc
            and prev_max_loc == max_loc
        ):
            break
        else:
            prev_min_val, prev_max_val, prev_min_loc, prev_max_loc = (
                min_val,
                max_val,
                min_loc,
                max_loc,
            )

        if max_val > threshold:
            # Prevent start_row, end_row, start_col, end_col be out of range of image
            start_row = max(0, max_loc[1] - template_h // 2)
            start_col = max(0, max_loc[0] - template_w // 2)
            end_row = min(template_result.shape[0], max_loc[1] + template_h // 2 + 1)
            end_col = min(template_result.shape[1], max_loc[0] + template_w // 2 + 1)

            template_result[start_row:end_row, start_col:end_col] = 0
            results.append(
                {
                    "max_val": max_val,
                    "xywh": (
                        max_loc[0],
                        max_loc[1],
                        max_loc[0] + template_w + 1,
                        max_loc[1] + template_h + 1,
                    ),
                }
            )

            # debug
            # src_dbg = cvtColor(src, COLOR_GRAY2BGR)
            # src_dbg = rectangle(
            #     src_dbg,
            #     (max_loc[0], max_loc[1]),
            #     (
            #         max_loc[0] + template_w + 1,
            #         max_loc[1] + template_h + 1,
            #     ),
            #     (0, 255, 0),
            #     thickness=3,
            # )
            # src_dbg = putText(
            #     src_dbg,
            #     f"{max_val:.5f}",
            #     (5, src_dbg.shape[0] - 5),
            #     FONT_HERSHEY_SIMPLEX,
            #     1,
            #     (0, 255, 0),
            #     thickness=2,
            # )
            # imshow("src_rect", src_dbg)
            # imshow("templ", template)
            # waitKey(750)
            # destroyAllWindows()

    return results
