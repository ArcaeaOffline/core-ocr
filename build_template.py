import base64
import json

import cv2
from src.arcaea_offline_ocr.template import load_digit_template

TEMPLATES = [
    ("GeoSansLight_Regular", "./assets/templates/GeoSansLightRegular.png"),
    ("GeoSansLight_Italic", "./assets/templates/GeoSansLightItalic.png"),
]

OUTPUT_FILE = "_builtin_templates.py"
output = ""

for name, file in TEMPLATES:
    template_res = load_digit_template(file)
    template_res_b64 = {
        key: base64.b64encode(cv2.imencode(".png", template_img)[1]).decode("utf-8")
        for key, template_img in template_res.items()
    }
    # jpg_as_text = base64.b64encode(buffer)
    output += f"{name} = {json.dumps(template_res_b64)}"
    output += "\n"

with open(OUTPUT_FILE, "w", encoding="utf-8") as of:
    of.write(output)
