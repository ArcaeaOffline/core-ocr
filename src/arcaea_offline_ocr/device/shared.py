import attrs


@attrs.define
class DeviceOcrResult:
    song_id: None
    title: None
    rating_class: int
    pure: int
    far: int
    lost: int
    score: int
    max_recall: int
    clear_type: None
