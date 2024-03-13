import cv2
import numpy as np
import matplotlib.pyplot as plt
import easyocr
from ultralytics import YOLO
from PIL import Image, ImageChops

import sys
import math
import yaml
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Load class names from data.yaml
class_names = []
with open('data.yaml', 'r') as file:
    data = yaml.load(file, Loader=yaml.FullLoader)
    class_names = data['names']

import argparse

parser = argparse.ArgumentParser(description='Process an input image.')
parser.add_argument('input_image', type=str, help='Path to the input image')
args = parser.parse_args()

input_image=args.input_image

# Load a model
model_path = 'best.pt'

model = YOLO(model_path)  # pretrained YOLOv8n model


img = cv2.imread(input_image)
results = model([img])

height, width, _ = img.shape

# Initialize a mask with all white pixels
mask = np.ones((height, width, 3), dtype=np.uint8) * 255


# Process results for the first image (assuming there's only one image in the batch)
for result in results:
    boxes = result.boxes  # Extract bounding box coordinates
    original_image = result.orig_img  # Extract the original image

    if boxes is not None:
        # Iterate over detected boxes
        for box, class_id, score in zip(boxes.xyxy, boxes.cls, boxes.conf):
            # Extract coordinates of the bounding box
            x1, y1, x2, y2 = box[:4].tolist()

            # Convert coordinates to integers
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Get the class name from class_names list
            class_name = class_names[int(class_id)]

            # Replace the cropped area in the mask with white
            mask[y1:y2, x1:x2] = 0

# Apply the mask to the original image to make everything except the cropped shapes white
result_image = np.where(mask == 255, 255, img)

img1 = Image.open(input_image)
result_image_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
cv2.imwrite("result1.png", result_image_bgr)
result1 = Image.open("result1.png")

diff1 = ImageChops.subtract(result1, img1)
# diff2 = ImageChops.subtract(result_image, img)
#
diff1.save("result.png")



# Create a reader to detect text
reader = easyocr.Reader(['en'])
image_path = 'result.png'
image = cv2.imread(image_path)

# Convert the image to gray scale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
results = reader.readtext(gray, paragraph=False)

# Create a mask to cover the texts
for (bbox, text, prob) in results:
    # Unpack the bounding box
    (top_left, top_right, bottom_right, bottom_left) = bbox
    top_left = (int(top_left[0]), int(top_left[1]))
    bottom_right = (int(bottom_right[0]), int(bottom_right[1]))

    # Draw a filled rectangle over the detected text area
    cv2.rectangle(gray, top_left, bottom_right, (0, 0, 0), -1)
    
# Apply edge detection
edges = cv2.Canny(gray, 50, 150, apertureSize=3)
_, thresholded = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

# Find contours in the thresholded image
contours, _ = cv2.findContours(thresholded, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# We can draw the contours on the image to visualize them
image_with_contours = image.copy()

# Draw the contours and put a small numbering (index) on each contour on the image
for i, cnt in enumerate(contours):
    # Draw each contour
    cv2.drawContours(img, [cnt], 0, (0, 0, 255), 2)
    
    # Calculate the center of the contour for placing the text
    M = cv2.moments(cnt)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        # This is the case for very small contours, where the number of pixels is too small for cv2.moments to work properly
        cX, cY = cnt[0][0]
    # Put the index number at the center of the contour
    cv2.putText(img, str(i), (cX - 10, cY + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# Convert the image to RGB for displaying correctly in the notebook
image_with_contours_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Display the image with contours and index numbers
plt.figure(figsize=(12, 8))
plt.imshow(image_with_contours_rgb)
plt.axis('off')  # Hide the axis
plt.show()

cv2.imwrite("line_output.jpg", image_with_contours_rgb)
