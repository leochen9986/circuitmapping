import yaml
import cv2
from ultralytics import YOLO
import os
import numpy as np
import json
import easyocr

# Convert all detected obj into file
class_names = []
with open('data.yaml', 'r') as file:
    data = yaml.load(file, Loader=yaml.FullLoader)
    class_names = data['names']

model_path = os.path.join('.', 'models', 'C:/Users/ngzek/PycharmProjects/object_detection/runs/detect/train6/weights/best.pt')

model = YOLO(model_path)  # pretrained YOLOv8n model

img_path = 'C:/Users/ngzek/PycharmProjects/Work/images/F882B1E5-117B-48CC-8F78-80D6306DCD16.png'  # Update this path to the actual location of your image file
#C:/Users/ngzek/PycharmProjects/Work/images/F66DF885-0877-4909-87D3-CCA55BDC507F.jpg
img = cv2.imread(img_path)
results = model([img])

# Create an OCR reader instance
reader = easyocr.Reader(['en'], gpu=False)  # 'en' for English language, gpu=False to run on CPU

box_id = 0
output = {"Elements": []}

expand_threshold = 30  # Adjust this value as needed


for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    original_image = result.orig_img  # Extract the original image
    if boxes is not None:
        # Iterate over detected boxes
        for box, class_id, score in zip(boxes.xyxy, boxes.cls, boxes.conf):
            box_id += 1
            class_name = class_names[int(class_id)]
            # Extract coordinates of the bounding box
            x1, y1, x2, y2 = map(int, box[:4])
            # Expanding the bounding box by the threshold
            expanded_x1 = max(0, x1 - expand_threshold)
            expanded_y1 = max(0, y1 - expand_threshold)
            expanded_x2 = min(original_image.shape[1], x2 + expand_threshold)
            expanded_y2 = min(original_image.shape[0], y2 + expand_threshold)

            # Cropping the image using the expanded bounding box
            cropped_image = original_image[expanded_y1:expanded_y2, expanded_x1:expanded_x2]

            ocr_result = reader.readtext(cropped_image)
            name = None
            for (bbox, text, prob) in ocr_result:
                if prob > 0.3:
                    name = text
                    break

            contained_elements = []
            # Check if bounding box is within another bounding box
            for other_box, other_class_id in zip(boxes.xyxy, boxes.cls):
                if not np.array_equal(box, other_box):
                    x1_other, y1_other, x2_other, y2_other = other_box[:4]
                    if (x1_other >= x1 and y1_other >= y1 and x2_other <= x2 and y2_other <= y2):
                        print("One bounding box is completely within another!")
                        # Add the contained object to the list
                        contained_elements.append({
                            "ID": box_id,
                            "name": name if name else f"B{class_id}",
                            "shape": class_names[int(other_class_id)],  # Assuming class names are same as shapes
                            "score": score.item()  # Convert the tensor score to a Python float
                        })

            # Write the ID and recognized name on the original image
            cv2.putText(original_image, f"{box_id}: {name}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            element = {
                "ID": box_id,
                "name": name if name else f"B{class_id}",
                "shape": class_name,
                "score": score.item(),  # Convert the tensor score to a Python float
                "ContainedElement": contained_elements
            }
            output["Elements"].append(element)

    # Display the image with IDs and names
    cv2.imshow('Image with IDs and Names', original_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Write output to JSON file
json_string = json.dumps(output, indent=4)
with open("detected_objects.json", "w") as json_file:
    json_file.write(json_string)

print(f"Output saved to detected_objects.json")
