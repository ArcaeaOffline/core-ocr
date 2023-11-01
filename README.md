# Arcaea Offline OCR

## Example

```py
import cv2
from arcaea_offline_ocr.device.ocr import DeviceOcr
from arcaea_offline_ocr.device.rois.definition import DeviceRoisAutoT2
from arcaea_offline_ocr.device.rois.extractor import DeviceRoisExtractor
from arcaea_offline_ocr.device.rois.masker import DeviceRoisMaskerAutoT2
from arcaea_offline_ocr.phash_db import ImagePhashDatabase

img_path = "/path/to/opencv/supported/image/formats.jpg"
img = cv2.imread(img_path, cv2.IMREAD_COLOR)

rois = DeviceRoisAutoT2(img.shape[1], img.shape[0])
extractor = DeviceRoisExtractor(img, rois)
masker = DeviceRoisMaskerAutoT2()

knn_model = cv2.ml.KNearest.load("/path/to/trained/knn/model.dat")
phash_db = ImagePhashDatabase("/path/to/image/phash/database.db")

ocr = DeviceOcr(extractor, masker, knn_model, phash_db)
print(ocr.ocr())
```

```sh
$ python example.py
DeviceOcrResult(rating_class=2, pure=1135, far=11, lost=0, score=9953016, max_recall=1146, song_id='ringedgenesis', song_id_possibility=0.953125, clear_status=2, partner_id='45', partner_id_possibility=0.8046875)
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

[283375/image-phash-database](https://github.com/283375/image-phash-database)
