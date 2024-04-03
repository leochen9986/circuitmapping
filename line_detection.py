import cv2
import numpy as np
import matplotlib.pyplot as plt
import easyocr
from ultralytics import YOLO
from PIL import Image, ImageChops
import os
import sys
import math
import yaml


def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# Function to find the farthest points in a contour
def find_farthest_points(cnt):
    max_dist = 0
    farthest_points = []
    for i in range(len(cnt)):
        for j in range(i + 1, len(cnt)):
            dist = np.linalg.norm(cnt[i][0] - cnt[j][0])
            if dist > max_dist:
                max_dist = dist
                farthest_points = [(cnt[i][0], cnt[j][0])]
            elif dist == max_dist:
                farthest_points.append((cnt[i][0], cnt[j][0]))
    return farthest_points

# Load class names from data.yaml
class_names = []
with open('data.yaml', 'r') as file:
    data = yaml.load(file, Loader=yaml.FullLoader)
    class_names = data['names']


def draw_contour_tips(img, contours):
    # Draw the contours and circles at the tips of each contour
    for cnt in contours:
        # Find the extreme points on the contour
        leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
        rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
        topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
        bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0])

        # Draw circles at these extreme points
        cv2.circle(img, leftmost, 5, (0, 255, 0), -1)
        cv2.circle(img, rightmost, 5, (0, 255, 0), -1)
        cv2.circle(img, topmost, 5, (0, 255, 0), -1)
        cv2.circle(img, bottommost, 5, (0, 255, 0), -1)
    return img

def find_contour_endpoints(contours):
    contour_endpoints = []
    for cnt in contours:
        if len(cnt) > 0:  # Check if the contour has any points
            cnt = cnt.squeeze()  # Remove single-dimensional entries from the shape of an array
            # Ensure cnt is 2-dimensional
            if cnt.ndim == 1:
                cnt = cnt.reshape(-1, 2)
            leftmost = cnt[cnt[:, 0].argmin()].tolist()
            rightmost = cnt[cnt[:, 0].argmax()].tolist()
            topmost = cnt[cnt[:, 1].argmin()].tolist()
            bottommost = cnt[cnt[:, 1].argmax()].tolist()
            endpoints = [leftmost, rightmost, topmost, bottommost]
            contour_endpoints.append((cnt, endpoints))
    return contour_endpoints

def get_connections(results,input_image):
    
    img = cv2.imread(input_image)    
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
    
    img1 = Image.open(input_image).convert('RGB')
    result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
    result1 = Image.fromarray(result_image_rgb)

    diff1 = ImageChops.subtract(result1, img1)

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
    
    return find_contour_endpoints(contours)
    