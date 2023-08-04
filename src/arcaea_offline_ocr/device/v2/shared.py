from cv2 import MORPH_CROSS, MORPH_ELLIPSE, MORPH_RECT, getStructuringElement

PFL_DENOISE_KERNEL = getStructuringElement(MORPH_RECT, [2, 2])
PFL_ERODE_KERNEL = getStructuringElement(MORPH_RECT, [3, 3])
PFL_CLOSE_HORIZONTAL_KERNEL = getStructuringElement(MORPH_RECT, [10, 1])

MAX_RECALL_DENOISE_KERNEL = getStructuringElement(MORPH_RECT, [3, 3])
MAX_RECALL_ERODE_KERNEL = getStructuringElement(MORPH_RECT, [2, 2])
MAX_RECALL_CLOSE_KERNEL = getStructuringElement(MORPH_RECT, [20, 1])
