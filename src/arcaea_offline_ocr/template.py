import pickle
from base64 import b64decode
from typing import Any, Dict, List, Literal, Tuple, TypedDict, Union

from cv2 import (
    CHAIN_APPROX_SIMPLE,
    COLOR_BGR2GRAY,
    COLOR_GRAY2BGR,
    FONT_HERSHEY_SIMPLEX,
    IMREAD_GRAYSCALE,
    RETR_EXTERNAL,
    THRESH_BINARY_INV,
    TM_CCOEFF_NORMED,
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
from numpy import ndarray

from ._builtin_templates import (
    DEFAULT_ITALIC,
    DEFAULT_ITALIC_ERODED,
    DEFAULT_REGULAR,
    DEFAULT_REGULAR_ERODED,
)
from .types import Mat

__all__ = [
    "TemplateItem",
    "DigitTemplate",
    "load_builtin_digit_template",
    "MatchTemplateMultipleResult",
    "matchTemplateMultiple",
]

# a list of Mat showing following characters:
# [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, ']
TemplateItem = Union[List[Mat], Tuple[Mat]]


class DigitTemplate:
    __slots__ = ["regular", "italic", "regular_eroded", "italic_eroded"]

    regular: TemplateItem
    italic: TemplateItem
    regular_eroded: TemplateItem
    italic_eroded: TemplateItem

    def __ensure_template_item(self, item):
        return (
            isinstance(item, (list, tuple))
            and len(item) == 11
            and all(isinstance(i, ndarray) for i in item)
        )

    def __init__(self, regular, italic, regular_eroded, italic_eroded):
        self.regular = regular
        self.italic = italic
        self.regular_eroded = regular_eroded
        self.italic_eroded = italic_eroded

    def __setattr__(self, __name: str, __value: Any):
        if __name in {
            "regular",
            "italic",
            "regular_eroded",
            "italic_eroded",
        } and self.__ensure_template_item(__value):
            super().__setattr__(__name, __value)
            return

        raise ValueError(
            "Invalid attribute set, expected type TemplateItem or invalid attribute name."
        )


def load_builtin_digit_template(name: Literal["default"]) -> DigitTemplate:
    CONSTANTS = {
        "default": [
            DEFAULT_REGULAR,
            DEFAULT_ITALIC,
            DEFAULT_REGULAR_ERODED,
            DEFAULT_ITALIC_ERODED,
        ]
    }
    args = CONSTANTS[name]
    args = [
        [pickle.loads(b64decode(encoded_str)) for encoded_str in arg] for arg in args
    ]

    return DigitTemplate(*args)


class MatchTemplateMultipleResult(TypedDict):
    max_val: float
    xywh: Tuple[int, int, int, int]


def matchTemplateMultiple(
    src: Mat, template: Mat, threshold: float = 0.1
) -> List[MatchTemplateMultipleResult]:
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
