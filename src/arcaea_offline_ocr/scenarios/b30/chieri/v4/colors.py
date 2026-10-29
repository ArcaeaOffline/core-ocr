import numpy as np

__all__ = [
    "BYD_MAX_HSV",
    "BYD_MIN_HSV",
    "FTR_MAX_HSV",
    "FTR_MIN_HSV",
    "PRS_MAX_HSV",
    "PRS_MIN_HSV",
]

BYD_MIN_HSV = np.array([158, 120, 0], np.uint8)
BYD_MAX_HSV = np.array([172, 255, 255], np.uint8)

FTR_MIN_HSV = np.array([145, 70, 0], np.uint8)
FTR_MAX_HSV = np.array([160, 255, 255], np.uint8)

PRS_MIN_HSV = np.array([45, 60, 0], np.uint8)
PRS_MAX_HSV = np.array([70, 255, 255], np.uint8)
