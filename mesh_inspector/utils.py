import cv2
import numpy as np
import json
import os
import pandas as pd

from mesh_django import settings

json_path = os.path.join(settings.BASE_DIR, 'data', 'tolerances.json')
with open(json_path) as f:
    TOLERANCES = json.load(f)

# with open(os.path.join("data", "tolerances.json")) as f:
#     TOLERANCES = json.load(f)

def check_tolerance(weight_class, lwo, swo):
    lwo_min, lwo_max = TOLERANCES[weight_class]["LWO"]
    swo_min, swo_max = TOLERANCES[weight_class]["SWO"]
    return lwo_min <= lwo <= lwo_max and swo_min <= swo <= swo_max

def process_image(image_path, weight_class, scale_pixels_per_inch):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    def is_diamond(cnt):
        approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
        return len(approx) == 4 and cv2.isContourConvex(approx)

    diamonds = [cnt for cnt in contours if is_diamond(cnt)]

    results = []
    for cnt in diamonds:
        x, y, w, h = cv2.boundingRect(cnt)
        lwo = round(h / scale_pixels_per_inch, 4)
        swo = round(w / scale_pixels_per_inch, 4)
        passed = check_tolerance(weight_class, lwo, swo)
        results.append({"LWO (in)": lwo, "SWO (in)": swo, "Pass": passed})
        color = (0, 255, 0) if passed else (0, 0, 255)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

    os.makedirs("outputs", exist_ok=True)
    out_img_path = os.path.join("outputs", f"annotated_{os.path.basename(image_path)}")
    cv2.imwrite(out_img_path, img)

    pd.DataFrame(results).to_csv(os.path.join("outputs", "measurements.csv"), index=False)
    return results, out_img_path, len(diamonds)
