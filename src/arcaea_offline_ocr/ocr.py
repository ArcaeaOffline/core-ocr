import re
from typing import Dict, List

import cv2
import numpy as np
from imutils import grab_contours
from imutils import resize as imutils_resize
from pytesseract import image_to_string

from .template import (
    MatchTemplateMultipleResult,
    TemplateItem,
    load_builtin_digit_template,
    matchTemplateMultiple,
)
from .types import Mat, cv2_ml_KNearest

__all__ = [
    "group_numbers",
    "FilterDigitResultDict",
    "filter_digit_results",
    "ocr_digits",
    "ocr_pure",
    "ocr_far_lost",
    "ocr_score",
    "ocr_max_recall",
    "ocr_rating_class",
    "ocr_title",
]


def group_numbers(numbers: List[int], threshold: int) -> List[List[int]]:
    """
    ```
    numbers = [26, 189, 303, 348, 32, 195, 391, 145, 77]
    group_numbers(numbers, 10) -> [[26, 32], [77], [145], [189, 195], [303], [348], [391]]
    group_numbers(numbers, 5) -> [[26], [32], [77], [145], [189], [195], [303], [348], [391]]
    group_numbers(numbers, 50) -> [[26, 32, 77], [145, 189, 195], [303, 348, 391]]
    # from Bing AI
    ```
    """
    numbers.sort()
    # Initialize an empty list of groups
    groups = []
    # Initialize an empty list for the current group
    group = []
    # Loop through the numbers
    for number in numbers:
        # If the current group is empty or the number is within the threshold of the last number in the group
        if not group or number - group[-1] <= threshold:
            # Append the number to the current group
            group.append(number)
        # Otherwise
        else:
            # Append the current group to the list of groups
            groups.append(group)
            # Start a new group with the number
            group = [number]
    # Append the last group to the list of groups
    groups.append(group)
    # Return the list of groups
    return groups


class FilterDigitResultDict(MatchTemplateMultipleResult):
    digit: int


def filter_digit_results(
    results: Dict[int, List[MatchTemplateMultipleResult]], threshold: int
):
    result_sorted_by_x_pos: Dict[
        int, List[FilterDigitResultDict]
    ] = {}  # Dictionary to store results sorted by x-position

    # Iterate over each digit and its match results
    for digit, match_results in results.items():
        if match_results:
            # Iterate over each match result
            for result in match_results:
                x_pos = result["xywh"][0]  # Extract x-position from result
                _dict = {**result, "digit": digit}  # Add digit information to result

                # Store result in result_sorted_by_x_pos dictionary
                if result_sorted_by_x_pos.get(x_pos) is None:
                    result_sorted_by_x_pos[x_pos] = [_dict]
                else:
                    result_sorted_by_x_pos[x_pos].append(_dict)

    x_poses_grouped: List[List[int]] = group_numbers(
        list(result_sorted_by_x_pos), threshold
    )  # Group x-positions based on threshold

    final_result: Dict[
        int, List[MatchTemplateMultipleResult]
    ] = {}  # Dictionary to store final filtered results

    # Iterate over each group of x-positions
    for x_poses in x_poses_grouped:
        possible_results = []
        # Iterate over each x-position in the group
        for x_pos in x_poses:
            # Retrieve all results associated with the x-position
            possible_results.extend(result_sorted_by_x_pos.get(x_pos, []))

        if possible_results:
            # Sort the results based on "max_val" in descending order and select the top result
            result = sorted(
                possible_results,
                key=lambda d: d["max_val"],
                reverse=True,
            )[0]
            result_digit = result["digit"]  # Get the digit value from the result
            result.pop("digit", None)  # Remove the digit key from the result

            # Store the result in the final_result dictionary
            if final_result.get(result_digit) is None:
                final_result[result_digit] = [result]
            else:
                final_result[result_digit].append(result)

    return final_result


def ocr_digits(
    img: Mat,
    templates: TemplateItem,
    template_threshold: float,
    filter_threshold: int,
):
    templates = dict(enumerate(templates[:10]))
    results: Dict[int, List[MatchTemplateMultipleResult]] = {}
    for digit, template in templates.items():
        template = imutils_resize(template, height=img.shape[0])
        results[digit] = matchTemplateMultiple(img, template, template_threshold)
    results = filter_digit_results(results, filter_threshold)
    result_x_digit_map = {}
    for digit, match_results in results.items():
        if match_results:
            for result in match_results:
                result_x_digit_map[result["xywh"][0]] = digit
    digits_sorted_by_x = dict(sorted(result_x_digit_map.items()))
    joined_str = "".join([str(digit) for digit in digits_sorted_by_x.values()])
    return int(joined_str) if joined_str else None


def ocr_digits_knn_model(img_gray: Mat, knn_model: cv2_ml_KNearest):
    if img_gray.shape[:2] != (20, 20):
        img = cv2.resize(img_gray, [20, 20])
    else:
        img = img_gray.copy()

    img = img.astype(np.float32)
    img = img.reshape([1, -1])
    retval, _, _, _ = knn_model.findNearest(img, 10)
    return int(retval)


def ocr_pure(img_masked: Mat):
    template = load_builtin_digit_template("default")
    return ocr_digits(
        img_masked, template.regular, template_threshold=0.6, filter_threshold=3
    )


def ocr_far_lost(img_masked: Mat):
    template = load_builtin_digit_template("default")
    return ocr_digits(
        img_masked, template.italic, template_threshold=0.6, filter_threshold=3
    )


def ocr_score(img_cropped: Mat):
    templates = load_builtin_digit_template("default").regular
    templates_dict = dict(enumerate(templates[:10]))

    cnts = cv2.findContours(
        img_cropped.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cnts = grab_contours(cnts)
    rects = [cv2.boundingRect(cnt) for cnt in cnts]
    rects = sorted(rects, key=lambda r: r[0])

    # debug
    # [rectangle(img, rect, (128, 128, 128), 2) for rect in rects]
    # imshow("img", img)
    # waitKey(0)

    result = ""
    for rect in rects:
        x, y, w, h = rect
        roi = img_cropped[y : y + h, x : x + w]

        digit_results: Dict[int, float] = {}
        for digit, template in templates_dict.items():
            template = cv2.resize(template, roi.shape[::-1])
            template_result = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(template_result)
            digit_results[digit] = max_val

        digit_results = {k: v for k, v in digit_results.items() if v > 0.5}
        if digit_results:
            best_match_digit = max(digit_results, key=digit_results.get)
            result += str(best_match_digit)
    return int(result) if result else None

    # return ocr_digits(
    #     img_cropped,
    #     template.regular,
    #     template_threshold=0.5,
    #     filter_threshold=10,
    # )


def ocr_max_recall(img_cropped: Mat):
    try:
        texts = image_to_string(img_cropped).split(" ")  # type: List[str]
        texts.reverse()
        for text in texts:
            if re.match(r"^[0-9]+$", text):
                return int(text)
    except Exception as e:
        return None


def ocr_rating_class(img_cropped: Mat):
    try:
        text = image_to_string(img_cropped)  # type: str
        text = text.lower()
        if "past" in text:
            return 0
        elif "present" in text:
            return 1
        elif "future" in text:
            return 2
        elif "beyond" in text:
            return 3
    except Exception as e:
        return None


def ocr_title(img_cropped: Mat):
    try:
        return image_to_string(img_cropped).replace("\n", "")
    except Exception as e:
        return ""
