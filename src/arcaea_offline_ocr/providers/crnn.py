from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import cv2
import numpy as np
from typing_extensions import override

from .base import OcrTextProvider

if TYPE_CHECKING:
    from collections.abc import Sequence

    import onnxruntime as ort
    from cv2.typing import MatLike

logger = logging.getLogger(__name__)


class OcrCrnnTextProvider(OcrTextProvider):
    DEFAULT_INFO_FILENAME: ClassVar[str] = "model_info.json"

    image_width: int
    image_height: int
    labels: list[str]
    blank_token: str
    pad_token: str
    session: ort.InferenceSession
    _input_name: str
    _output_index: int

    def __init__(self, model_path: str | Path, info_path: str | Path | None = None):
        model_path = Path(model_path)
        if info_path is None:
            info_path = model_path.parent / self.DEFAULT_INFO_FILENAME

        info = json.loads(Path(info_path).read_text(encoding="utf-8"))["training"]
        self.image_width = int(info["image_width"])
        self.image_height = int(info["image_height"])
        self.labels = list(info["labels"])
        self.blank_token = info["blank_token"]
        self.pad_token = info["pad_token"]

        # lazy import only when needed
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(model_path))
        self._input_name = str(self.session.get_inputs()[0].name)
        output_names = [str(output.name) for output in self.session.get_outputs()]
        # prefer the model's pre-argmax `decoded_output` if present
        self._output_index = (
            output_names.index("decoded_output")
            if "decoded_output" in output_names
            else 0
        )

    def _decode(self, predictions: Sequence[int]) -> str:
        chars = [self.labels[i] for i in predictions]

        # CTC collapse: drop consecutive repeats and blanks
        decoded_chars: list[str] = []
        last_char: str | None = None
        for char in chars:
            if char == last_char:
                continue
            if char == self.blank_token:
                last_char = None
                continue
            decoded_chars.append(char)
            last_char = char

        # strip pad tokens, stop at two consecutive pads
        result: list[str] = []
        placeholder_count = 0
        for char in decoded_chars:
            if placeholder_count == 2:
                break
            if char == self.pad_token:
                placeholder_count += 1
                continue
            placeholder_count = 0
            result.append(char)

        return "".join(result)

    @override
    def result_raw(self, img: MatLike, /) -> str | None:
        """
        :param img: BGR format roi
        """
        try:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (self.image_width, self.image_height))
            # the model takes a single (height, width, 3) uint8 image, no batch dim
            tensor = resized.astype(np.uint8)

            outputs = self.session.run(None, {self._input_name: tensor})
            # onnxruntime is untyped; the decoded output is an int index array
            predictions: Sequence[int] = (
                np.asarray(outputs[self._output_index]).ravel().tolist()
            )
            return self._decode(predictions)
        except Exception:
            logger.exception("Error occurred during CRNN OCR")
            return None

    @override
    def result(self, img: MatLike, /) -> str | None:
        """
        :param img: BGR format roi
        """
        return self.result_raw(img)
