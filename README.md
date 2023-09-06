# Arcaea Offline OCR

## Example

```py
from arcaea_offline_ocr.device.v2.rois import DeviceV2AutoRois
from arcaea_offline_ocr.device.v2.ocr import DeviceV2Ocr
from arcaea_offline_ocr.sift_db import SIFTDatabase
from arcaea_offline_ocr.utils import imread_unicode
import cv2

knn_model = cv2.ml.KNearest_load(r'/path/to/knn/model')
sift_db = SIFTDatabase(r'/path/to/sift/database.db')

rois = DeviceV2AutoRois(imread_unicode(r'/path/to/your/screenshot.jpg'))  # any format that opencv-python supports
ocr = DeviceV2Ocr(knn_model, sift_db)
result = ocr.ocr(rois)
print(result)
```

```sh
$ python example.py
DeviceOcrResult(rating_class=2, pure=1371, far=62, lost=34, score=9558078, max_recall=330, song_id='abstrusedilemma', title=None, clear_type=None)
```

## Credits

[283375/image-sift-database](https://github.com/283375/image-sift-database)
