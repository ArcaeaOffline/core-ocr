import base64
import json
import pickle

import cv2
import imutils
import numpy
from imutils import contours


def load_template_image(filename: str) -> dict[int, cv2.Mat]:
    """
    Arguments:
        filename -- An image with white background and black "0 1 2 3 4 5 6 7 8 9 '" text.
    """
    # https://pyimagesearch.com/2017/07/17/credit-card-ocr-with-opencv-and-python/
    ref = cv2.imread(filename)
    ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    ref = cv2.threshold(ref, 10, 255, cv2.THRESH_BINARY_INV)[1]
    refCnts = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    refCnts = imutils.grab_contours(refCnts)
    refCnts = contours.sort_contours(refCnts, method="left-to-right")[0]
    digits = {}
    keys = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, "'"]
    for key, cnt in zip(keys, refCnts):
        (x, y, w, h) = cv2.boundingRect(cnt)
        roi = ref[y : y + h, x : x + w]
        digits[key] = roi
    return list(digits.values())


def process_default(img_path: str):
    template_res = load_template_image(img_path)
    template_res_pickled = [
        base64.b64encode(
            pickle.dumps(template_arr, protocol=pickle.HIGHEST_PROTOCOL)
        ).decode("utf-8")
        for template_arr in template_res
    ]
    return json.dumps(template_res_pickled)


def process_eroded(img_path: str):
    kernel = numpy.ones((5, 5), numpy.uint8)
    template_res = load_template_image(img_path)
    template_res_eroded = []
    # cv2.imshow("orig", template_res[7])
    for template in template_res:
        # add borders
        template = cv2.copyMakeBorder(
            template, 10, 10, 10, 10, cv2.BORDER_CONSTANT, None, (0, 0, 0)
        )
        # erode
        template = cv2.erode(template, kernel)
        # remove borders
        h, w = template.shape
        template = template[10 : h - 10, 10 : w - 10]
        template_res_eroded.append(template)
    # cv2.imshow("erode", template_res_eroded[7])
    # cv2.waitKey(0)
    template_res_pickled = [
        base64.b64encode(
            pickle.dumps(template_arr, protocol=pickle.HIGHEST_PROTOCOL)
        ).decode("utf-8")
        for template_arr in template_res_eroded
    ]
    return json.dumps(template_res_pickled)


TEMPLATES = [
    (
        "DEFAULT_REGULAR",
        "./assets/templates/GeoSansLightRegular.png",
        process_default,
    ),
    (
        "DEFAULT_ITALIC",
        "./assets/templates/GeoSansLightItalic.png",
        process_default,
    ),
    (
        "DEFAULT_REGULAR_ERODED",
        "./assets/templates/GeoSansLightRegular.png",
        process_eroded,
    ),
    (
        "DEFAULT_ITALIC_ERODED",
        "./assets/templates/GeoSansLightItalic.png",
        process_eroded,
    ),
]

OUTPUT_FILE = "_builtin_templates.py"
output = ""

for name, img_path, process_func in TEMPLATES:
    output += f"{name} = {process_func(img_path)}"
    output += "\n"

with open(OUTPUT_FILE, "w", encoding="utf-8") as of:
    of.write(output)
