from ....types import Mat


class DeviceRoisMasker:
    @classmethod
    def pure(cls, roi_bgr: Mat) -> Mat:
        raise NotImplementedError()

    @classmethod
    def far(cls, roi_bgr: Mat) -> Mat:
        raise NotImplementedError()

    @classmethod
    def lost(cls, roi_bgr: Mat) -> Mat:
        raise NotImplementedError()

    @classmethod
    def score(cls, roi_bgr: Mat) -> Mat:
        raise NotImplementedError()

    @classmethod
    def rating_class_pst(cls, roi_bgr: Mat) -> Mat:
        raise NotImplementedError()

    @classmethod
    def rating_class_prs(cls, roi_bgr: Mat) -> Mat:
        raise NotImplementedError()

    @classmethod
    def rating_class_ftr(cls, roi_bgr: Mat) -> Mat:
        raise NotImplementedError()

    @classmethod
    def rating_class_byd(cls, roi_bgr: Mat) -> Mat:
        raise NotImplementedError()

    @classmethod
    def rating_class_etr(cls, roi_bgr: Mat) -> Mat:
        raise NotImplementedError()

    @classmethod
    def max_recall(cls, roi_bgr: Mat) -> Mat:
        raise NotImplementedError()

    @classmethod
    def clear_status_track_lost(cls, roi_bgr: Mat) -> Mat:
        raise NotImplementedError()

    @classmethod
    def clear_status_track_complete(cls, roi_bgr: Mat) -> Mat:
        raise NotImplementedError()

    @classmethod
    def clear_status_full_recall(cls, roi_bgr: Mat) -> Mat:
        raise NotImplementedError()

    @classmethod
    def clear_status_pure_memory(cls, roi_bgr: Mat) -> Mat:
        raise NotImplementedError()
