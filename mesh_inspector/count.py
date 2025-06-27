import json
import cv2
import numpy as np
import os

from mesh_django import settings

json_path = os.path.join(settings.BASE_DIR, 'data', 'tolerances.json')
with open(json_path) as f:
    TOLERANCES = json.load(f)

def check_tolerance(weight_class, lwo, swo):
    lwo_min, lwo_max = TOLERANCES[weight_class]["LWO"]
    swo_min, swo_max = TOLERANCES[weight_class]["SWO"]
    return lwo_min <= lwo <= lwo_max and swo_min <= swo <= swo_max

def is_diamond(cnt, epsilon_ratio=0.05, aspect_ratio_range=(0.5, 1.5), min_area=80):
    """Return True if the contour is likely a diamond shape."""
    approx = cv2.approxPolyDP(cnt, epsilon_ratio * cv2.arcLength(cnt, True), True)
    if len(approx) >= 4 and cv2.isContourConvex(approx):
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        area = cv2.contourArea(cnt)
        return (
            aspect_ratio_range[0] <= aspect_ratio <= aspect_ratio_range[1] and
            area >= min_area
        )
    return False

def detect_and_annotate_diamonds(image_path, weight_class, scale_pixels_per_inch):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    orig = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Adjusted Canny thresholds for better edge detection
    edges = cv2.Canny(blur, 30, 100)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Optional: Save debug contours image
    debug_img = orig.copy()
    for c in contours:
        cv2.drawContours(debug_img, [c], -1, (255, 0, 0), 1)
    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite("outputs/debug_all_contours.jpg", debug_img)

    # Filter diamond-like contours
    diamonds = [cnt for cnt in contours if is_diamond(cnt)]

    # Sort for consistent numbering: top to bottom, then left to right
    diamonds.sort(key=lambda c: (cv2.boundingRect(c)[1], cv2.boundingRect(c)[0]))

    # Annotate detected diamonds
    annotated = img.copy()
    for i, cnt in enumerate(diamonds, 1):
        x, y, w, h = cv2.boundingRect(cnt)
        cx, cy = x + w // 2, y + h // 2
        cv2.drawContours(annotated, [cnt], -1, (0, 0, 255), 2)
        cv2.putText(annotated, str(i), (cx - 10, cy + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Draw total count at top-center
    count = len(diamonds)
    cv2.putText(annotated, f"Total: {count}",
                (int(img.shape[1] / 2) - 100, 60),
                cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 200), 4)

    # Save final output
    out_path = os.path.join("outputs", f"annotated_{os.path.basename(image_path)}")
    cv2.imwrite(out_path, annotated)

    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # def is_diamond(cnt):
    #     approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
    #     return len(approx) == 4 and cv2.isContourConvex(approx)

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

    return results,count, out_path

# --- Run ---
if __name__ == "__main__":
    image_path = "1234.jpeg"  # Replace with your file name if different
    count, output_image = detect_and_annotate_diamonds(image_path)
    print(f"✅ Detected {count} diamonds")
    print(f"📸 Annotated image saved to: {output_image}")
