from abc import ABC, abstractmethod

from cv2.typing import MatLike


class DeviceRoisMasker(ABC):
    @classmethod
    @abstractmethod
    def pure(cls, roi_bgr: MatLike) -> MatLike: ...

    @classmethod
    @abstractmethod
    def far(cls, roi_bgr: MatLike) -> MatLike: ...

    @classmethod
    @abstractmethod
    def lost(cls, roi_bgr: MatLike) -> MatLike: ...

    @classmethod
    @abstractmethod
    def score(cls, roi_bgr: MatLike) -> MatLike: ...

    @classmethod
    @abstractmethod
    def rating_class_pst(cls, roi_bgr: MatLike) -> MatLike: ...

    @classmethod
    @abstractmethod
    def rating_class_prs(cls, roi_bgr: MatLike) -> MatLike: ...

    @classmethod
    @abstractmethod
    def rating_class_ftr(cls, roi_bgr: MatLike) -> MatLike: ...

    @classmethod
    @abstractmethod
    def rating_class_byd(cls, roi_bgr: MatLike) -> MatLike: ...

    @classmethod
    @abstractmethod
    def rating_class_etr(cls, roi_bgr: MatLike) -> MatLike: ...

    @classmethod
    @abstractmethod
    def max_recall(cls, roi_bgr: MatLike) -> MatLike: ...

    @classmethod
    @abstractmethod
    def clear_status_track_lost(cls, roi_bgr: MatLike) -> MatLike: ...

    @classmethod
    @abstractmethod
    def clear_status_track_complete(cls, roi_bgr: MatLike) -> MatLike: ...

    @classmethod
    @abstractmethod
    def clear_status_full_recall(cls, roi_bgr: MatLike) -> MatLike: ...

    @classmethod
    @abstractmethod
    def clear_status_pure_memory(cls, roi_bgr: MatLike) -> MatLike: ...
