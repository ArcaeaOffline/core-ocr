import cv2


class Masker:
    @staticmethod
    def pure(roi_bgr: cv2.Mat) -> cv2.Mat:
        raise NotImplementedError()

    @staticmethod
    def far(roi_bgr: cv2.Mat) -> cv2.Mat:
        raise NotImplementedError()

    @staticmethod
    def lost(roi_bgr: cv2.Mat) -> cv2.Mat:
        raise NotImplementedError()

    @staticmethod
    def score(roi_bgr: cv2.Mat) -> cv2.Mat:
        raise NotImplementedError()

    @staticmethod
    def rating_class_pst(roi_bgr: cv2.Mat) -> cv2.Mat:
        raise NotImplementedError()

    @staticmethod
    def rating_class_prs(roi_bgr: cv2.Mat) -> cv2.Mat:
        raise NotImplementedError()

    @staticmethod
    def rating_class_ftr(roi_bgr: cv2.Mat) -> cv2.Mat:
        raise NotImplementedError()

    @staticmethod
    def rating_class_byd(roi_bgr: cv2.Mat) -> cv2.Mat:
        raise NotImplementedError()

    @staticmethod
    def max_recall(roi_bgr: cv2.Mat) -> cv2.Mat:
        raise NotImplementedError()

    @staticmethod
    def clear_status_track_lost(roi_bgr: cv2.Mat) -> cv2.Mat:
        raise NotImplementedError()

    @staticmethod
    def clear_status_track_complete(roi_bgr: cv2.Mat) -> cv2.Mat:
        raise NotImplementedError()

    @staticmethod
    def clear_status_full_recall(roi_bgr: cv2.Mat) -> cv2.Mat:
        raise NotImplementedError()

    @staticmethod
    def clear_status_pure_memory(roi_bgr: cv2.Mat) -> cv2.Mat:
        raise NotImplementedError()
