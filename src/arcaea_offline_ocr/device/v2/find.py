from typing import List, Tuple

import attrs
import cv2
import numpy as np

from ...crop import crop_xywh
from ...mask import mask_gray
from ...types import Mat, XYWHRect
from .definition import DeviceV2
from .shared import *


@attrs.define(kw_only=True)
class FindOcrBoundingRectsResult:
    pure: XYWHRect
    far: XYWHRect
    lost: XYWHRect
    max_recall: XYWHRect
    gray_masked_image: Mat


def find_ocr_bounding_rects(__img_bgr: Mat, device: DeviceV2):
    """
    [DEPRECATED]
    ---
    Deprecated since new method supports directly calculate rois.
    """

    img_masked = mask_gray(__img_bgr)

    # process pure/far/lost
    pfl_roi = crop_xywh(img_masked, device.pure_far_lost)
    # close small gaps in fonts
    # pfl_roi = cv2.GaussianBlur(pfl_roi, [5, 5], 0, 0)
    # cv2.imshow("test2", pfl_roi)
    # cv2.waitKey(0)

    pfl_roi = cv2.morphologyEx(pfl_roi, cv2.MORPH_OPEN, PFL_DENOISE_KERNEL)
    pfl_roi = cv2.morphologyEx(pfl_roi, cv2.MORPH_CLOSE, PFL_CLOSE_HORIZONTAL_KERNEL)

    pfl_contours, _ = cv2.findContours(
        pfl_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    pfl_contours = sorted(pfl_contours, key=cv2.contourArea)

    # pfl_roi_cnt = cv2.drawContours(pfl_roi, pfl_contours, -1, [50], 2)
    # cv2.imshow("test2", pfl_roi_cnt)
    # cv2.waitKey(0)

    pfl_rects = [list(cv2.boundingRect(c)) for c in pfl_contours]

    # for r in pfl_rects:
    #     img = pfl_roi.copy()
    #     cv2.imshow("test2", cv2.rectangle(img, r, [80] * 3, 2))
    #     cv2.waitKey(0)

    # only keep those rect.height > mask.height * 0.15
    pfl_rects = list(filter(lambda rect: rect[3] > pfl_roi.shape[0] * 0.15, pfl_rects))
    # choose the first 3 rects by rect.x value
    pfl_rects = sorted(pfl_rects, key=lambda rect: rect[0])[:3]
    # and sort them by rect.y
    # ensure it is pure -> far -> lost roi.
    pure_rect, far_rect, lost_rect = sorted(pfl_rects, key=lambda rect: rect[1])

    # for r in [pure_rect, far_rect, lost_rect]:
    #     img = pfl_roi.copy()
    #     cv2.imshow("test2", cv2.rectangle(img, r, [80] * 3, 2))
    #     cv2.waitKey(0)

    # process max recall
    max_recall_roi = crop_xywh(img_masked, device.max_recall_rating_class)
    max_recall_roi = cv2.morphologyEx(
        max_recall_roi, cv2.MORPH_OPEN, MAX_RECALL_DENOISE_KERNEL
    )
    max_recall_roi = cv2.erode(max_recall_roi, MAX_RECALL_ERODE_KERNEL)
    max_recall_roi = cv2.morphologyEx(
        max_recall_roi, cv2.MORPH_CLOSE, MAX_RECALL_CLOSE_KERNEL
    )
    max_recall_contours, _ = cv2.findContours(
        max_recall_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    max_recall_rects = [list(cv2.boundingRect(c)) for c in max_recall_contours]
    # only keep those rect.height > mask.height * 0.1
    max_recall_rects = list(
        filter(lambda rect: rect[3] > max_recall_roi.shape[0] * 0.1, max_recall_rects)
    )
    # select the 2nd rect by rect.x
    max_recall_rect = max_recall_rects[1]

    # img = max_recall_roi.copy()
    # cv2.imshow("test2", cv2.rectangle(img, max_recall_rect, [80] * 3, 2))
    # cv2.waitKey(0)

    # finally, map rect geometries to the original image
    for rect in [pure_rect, far_rect, lost_rect]:
        rect[0] += device.pure_far_lost[0]
        rect[1] += device.pure_far_lost[1]

    for rect in [max_recall_rect]:
        rect[0] += device.max_recall_rating_class[0]
        rect[1] += device.max_recall_rating_class[1]

    # add a 2px border to every rect
    for rect in [pure_rect, far_rect, lost_rect, max_recall_rect]:
        # width += 2, height += 2
        rect[2] += 4
        rect[3] += 4
        # top -= 1, left -= 1
        rect[0] -= 2
        rect[1] -= 2

    return FindOcrBoundingRectsResult(
        pure=XYWHRect(*pure_rect),
        far=XYWHRect(*far_rect),
        lost=XYWHRect(*lost_rect),
        max_recall=XYWHRect(*max_recall_rect),
        gray_masked_image=img_masked,
    )


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


def find_digits(__img_masked: Mat) -> List[Mat]:
    img_denoised = find_digits_preprocess(__img_masked)

    cv2.imshow("den", img_denoised)
    cv2.waitKey(0)

    contours, _ = cv2.findContours(
        img_denoised, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    img_x_roi = []  # type: List[Tuple[int, Mat]]
    # img_x_roi = list[tuple[int, Mat]] - list[tuple[rect.x, roi_denoised]]
    for contour in contours:
        rect = cv2.boundingRect(contour)
        # filter out rect.height < img.height * factor
        if rect[3] < img_denoised.shape[0] * 0.8:
            continue
        contour -= (rect[0], rect[1])
        img_denoised_roi = crop_xywh(img_denoised, rect)
        # make a same size black image
        contour_mask = np.zeros(img_denoised_roi.shape, img_denoised_roi.dtype)
        # fill the contour area with white pixels
        contour_mask = cv2.fillPoly(contour_mask, [contour], [255])
        # apply mask to cropped images
        img_denoised_roi_masked = cv2.bitwise_and(contour_mask, img_denoised_roi)
        img_x_roi.append((rect[0], img_denoised_roi_masked))

    # sort by rect.x
    img_x_roi = sorted(img_x_roi, key=lambda item: item[0])

    return [item[1] for item in img_x_roi]
