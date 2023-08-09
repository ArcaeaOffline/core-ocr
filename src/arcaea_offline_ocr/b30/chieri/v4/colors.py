import numpy as np

__all__ = [
    "FONT_THRESHOLD",
    "PURE_BG_MIN_HSV",
    "PURE_BG_MAX_HSV",
    "FAR_BG_MIN_HSV",
    "FAR_BG_MAX_HSV",
    "LOST_BG_MIN_HSV",
    "LOST_BG_MAX_HSV",
    "BYD_MIN_HSV",
    "BYD_MAX_HSV",
    "FTR_MIN_HSV",
    "FTR_MAX_HSV",
    "PRS_MIN_HSV",
    "PRS_MAX_HSV",
]

FONT_THRESHOLD = 160

PURE_BG_MIN_HSV = np.array([95, 140, 150], np.uint8)
PURE_BG_MAX_HSV = np.array([110, 255, 255], np.uint8)

FAR_BG_MIN_HSV = np.array([15, 100, 150], np.uint8)
FAR_BG_MAX_HSV = np.array([20, 255, 255], np.uint8)

LOST_BG_MIN_HSV = np.array([115, 60, 150], np.uint8)
LOST_BG_MAX_HSV = np.array([140, 255, 255], np.uint8)

BYD_MIN_HSV = (158, 120, 0)
BYD_MAX_HSV = (172, 255, 255)

FTR_MIN_HSV = (145, 70, 0)
FTR_MAX_HSV = (160, 255, 255)

PRS_MIN_HSV = (45, 60, 0)
PRS_MAX_HSV = (70, 255, 255)
