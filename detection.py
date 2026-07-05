"""
==========================================
Project: Object Detection using OpenCV
Algorithm: HOG Descriptor + Non-Maximum Suppression
Author: Tegnoor Singh
==========================================

Description:
This program detects people in an image using OpenCV's
HOG Descriptor and removes duplicate detections using
Non-Maximum Suppression (NMS).
"""

# ==========================
# Import Required Libraries
# ==========================

import cv2
import os
import numpy as np
from imutils.object_detection import non_max_suppression

# ==========================
# Load Input Image
# ==========================

image_path = "images/object detection.png"

image = cv2.imread(image_path)

if image is None:
    print("Error: Image not found!")
    print("Please check the image path.")
    exit()

# ==========================
# Resize Image
# ==========================

image = cv2.resize(image, (800, 600))

# ==========================
# Initialize HOG Descriptor
# ==========================

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# ==========================
# Detect People
# ==========================

(rects, weights) = hog.detectMultiScale(
    image,
    winStride=(8, 8),
    padding=(8, 8),
    scale=1.05
)

# ==========================
# Convert to NumPy Array
# ==========================

boxes = np.array(
    [[x, y, x + w, y + h] for (x, y, w, h) in rects]
)

# ==========================
# Apply Non-Maximum Suppression
# ==========================

if len(boxes) > 0:
    boxes = non_max_suppression(
        boxes,
        probs=None,
        overlapThresh=0.65
    )

# ==========================
# Draw Bounding Boxes
# ==========================

for (xA, yA, xB, yB) in boxes:

    cv2.rectangle(
        image,
        (xA, yA),
        (xB, yB),
        (0, 255, 0),
        2
    )

    cv2.putText(
        image,
        "Person",
        (xA, yA - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

# ==========================
# Display Number of People
# ==========================

cv2.putText(
    image,
    f"People Detected: {len(boxes)}",
    (20, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 0, 255),
    2
)

# ==========================
# Console Output
# ==========================

print("=" * 45)
print("      OBJECT DETECTION COMPLETED")
print("=" * 45)
print(f"People Detected : {len(boxes)}")
print("=" * 45)

# ==========================
# Save Output Image
# ==========================

os.makedirs("outputs", exist_ok=True)

output_path = "outputs/result.jpg"

cv2.imwrite(output_path, image)

print(f"Output image saved at: {output_path}")

# ==========================
# Display Output
# ==========================

cv2.imshow("Object Detection using OpenCV", image)

cv2.waitKey(0)
cv2.destroyAllWindows()
