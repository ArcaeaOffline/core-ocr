# Arcaea Offline OCR

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

## Example

> Results from `arcaea_offline_ocr 0.1.0a2`

### Build an image hash database (ihdb)

```py
import sqlite3
from pathlib import Path

import cv2

from arcaea_offline_ocr.builders.ihdb import (
    ImageHashDatabaseBuildTask,
    ImageHashesDatabaseBuilder,
)
from arcaea_offline_ocr.providers import ImageCategory, ImageHashDatabaseIdProvider
from arcaea_offline_ocr.scenarios.device import DeviceScenario

def build():
    def _read_partner_icon(image_path: str):
        return DeviceScenario.preprocess_char_icon(
            cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY),
        )

    builder = ImageHashesDatabaseBuilder()
    tasks = [
        ImageHashDatabaseBuildTask(
            image_path=str(file),
            image_id=file.stem,
            category=ImageCategory.JACKET,
        )
        for file in Path("/path/to/some/jackets").glob("*.jpg")
    ]

    tasks.extend(
        [
            ImageHashDatabaseBuildTask(
                image_path=str(file),
                image_id=file.stem,
                category=ImageCategory.PARTNER_ICON,
                imread_function=_read_partner_icon,
            )
            for file in Path("/path/to/some/partner_icons").glob("*.png")
        ],
    )

    with sqlite3.connect("/path/to/ihdb-X.Y.Z.db") as conn:
        builder.build(conn, tasks)
```

### Device OCR

```py
import json
import sqlite3
from dataclasses import asdict

import cv2

from arcaea_offline_ocr.providers import (
    ImageHashDatabaseIdProvider,
    OcrKNearestTextProvider,
)
from arcaea_offline_ocr.scenarios.device import (
    DeviceRoisAutoT2,
    DeviceRoisExtractor,
    DeviceRoisMaskerAutoT2,
    DeviceScenario,
)


with sqlite3.connect("/path/to/ihdb-X.Y.Z.db") as conn:
    img = cv2.imread("/path/to/your/screenshot.jpg")
    h, w = img.shape[:2]

    r = DeviceRoisAutoT2(w, h)
    m = DeviceRoisMaskerAutoT2()
    e = DeviceRoisExtractor(img, r)

    scenario = DeviceScenario(
        extractor=e,
        masker=m,
        knn_provider=OcrKNearestTextProvider(
            cv2.ml.KNearest.load("/path/to/knn_model.dat"),
        ),
        image_id_provider=ImageHashDatabaseIdProvider(conn),
    )
    result = scenario.result()

    with open("result.jsonc", "w", encoding="utf-8") as jf:
        json.dump(asdict(result), jf, indent=2, ensure_ascii=False)
```

```jsonc
// result.json
{
  "song_id": "vector",
  "rating_class": 1,
  "score": 9990996,
  "song_id_results": [
    {
      "image_id": "vector",
      "category": 0,
      "confidence": 1.0,
      "image_hash_type": 0
    },
    {
      "image_id": "clotho",
      "category": 0,
      "confidence": 0.71875,
      "image_hash_type": 0
    }
    // 28 more results omitted…
  ],
  "partner_id_results": [
    {
      "image_id": "23",
      "category": 1,
      "confidence": 0.90625,
      "image_hash_type": 0
    },
    {
      "image_id": "45",
      "category": 1,
      "confidence": 0.8828125,
      "image_hash_type": 0
    }
    // 28 more results omitted…
  ],
  "pure": 1000,
  "pure_inaccurate": null,
  "pure_early": null,
  "pure_late": null,
  "far": 2,
  "far_inaccurate": null,
  "far_early": null,
  "far_late": null,
  "lost": 0,
  "played_at": null,
  "max_recall": 1002,
  "clear_status": 2,
  "clear_type": null,
  "modifier": null
}
```

## License

This file is part of arcaea-offline-ocr, as called "This program" below.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

## Credits

- [JohannesBuchner/imagehash](https://github.com/JohannesBuchner/imagehash): `arcaea_offline_ocr.core.hashers` implementations reference
