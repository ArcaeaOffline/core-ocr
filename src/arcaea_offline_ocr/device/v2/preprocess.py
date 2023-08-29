import cv2

from ...types import Mat
from .shared import *


def find_digits_preprocess(__img_masked: Mat) -> Mat:
    img = __img_masked.copy()
    img_denoised = cv2.morphologyEx(img, cv2.MORPH_OPEN, PFL_DENOISE_KERNEL)
    # img_denoised = cv2.bitwise_and(img, img_denoised)

    denoise_contours, _ = cv2.findContours(
        img_denoised, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    # cv2.drawContours(img_denoised, contours, -1, [128], 2)

    # fill all contour.area < max(contour.area) * ratio with black pixels
    # for denoise purposes

    # define threshold contour area
    # we assume the smallest digit "1", is 80% height of the image,
    # and at least 1.5 pixel wide, considering cv2.contourArea always
    # returns a smaller value than the actual contour area.
    max_contour_area = __img_masked.shape[0] * 0.8 * 1.5
    filtered_contours = list(
        filter(lambda c: cv2.contourArea(c) >= max_contour_area, denoise_contours)
    )

    filtered_contours_flattened = {tuple(c.flatten()) for c in filtered_contours}

    for contour in denoise_contours:
        if tuple(contour.flatten()) not in filtered_contours_flattened:
            img_denoised = cv2.fillPoly(img_denoised, [contour], [0])

    # old algorithm, finding the largest contour area
    ## contour_area_tuples = [(contour, cv2.contourArea(contour)) for contour in contours]
    ## contour_area_tuples = sorted(
    ##     contour_area_tuples, key=lambda item: item[1], reverse=True
    ## )
    ## max_contour_area = contour_area_tuples[0][1]
    ## print(max_contour_area, [item[1] for item in contour_area_tuples])
    ## contours_filter_end_index = len(contours)
    ## for i, item in enumerate(contour_area_tuples):
    ##     contour, area = item
    ##     if area < max_contour_area * 0.15:
    ##         contours_filter_end_index = i
    ##         break
    ## contours = [item[0] for item in contour_area_tuples]
    ## for contour in contours[-contours_filter_end_index - 1:]:
    ##     img = cv2.fillPoly(img, [contour], [0])
    ##     img_denoised = cv2.fillPoly(img_denoised, [contour], [0])
    ## contours = contours[:contours_filter_end_index]

    return img_denoised
