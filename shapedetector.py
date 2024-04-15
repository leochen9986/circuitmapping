import yaml
import argparse
import cv2
from ultralytics import YOLO
import os
import numpy as np
import json
import easyocr
from scipy.optimize import linear_sum_assignment
import line_detection
import random

def point_in_box(point, box):
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2

#assign random color for visualization
def random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


def min_distance_point_rect(point, rect):
    px, py = point
    x1, y1, x2, y2 = rect
    # Check if the point is inside the rectangle
    if x1 <= px <= x2 and y1 <= py <= y2:
        return 0
    # Calculate distances to each edge
    distances = [
        abs(px - x1), abs(px - x2),  # Left and right edges
        abs(py - y1), abs(py - y2)   # Top and bottom edges
    ]
    # Return the minimum distance
    return min(distances)

# Convert all detected obj into file
class_names = []
with open('data.yaml', 'r') as file:
    data = yaml.load(file, Loader=yaml.FullLoader)
    class_names = data['names']

#Load YOLO Model
model_path =  'best.pt'
model = YOLO(model_path)  # pretrained YOLOv8n model


# Parse command-line arguments
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('img_path', type=str, help='Path to the image file')
args = parser.parse_args()

#Input Image
img_path = args.img_path  # Update this path to the actual location of your image file
img = cv2.imread(img_path)

#Yolo Processing
results = model([img])

#Wire Connection Detection
contour_endpoints = line_detection.get_connections(results,img_path)

# Create an OCR reader instance
reader = easyocr.Reader(['en'], gpu=True)  # 'en' for English language, gpu=False to run on CPU

box_id = 0
output = {"Elements": []}

expand_threshold = 0  # Adjust this value as needed


mask = np.ones_like(img) * 255
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    original_image = result.orig_img  # Extract the original image
    if boxes is not None:
        # Iterate over detected boxes and apply white mask
        for box, class_id, score in zip(boxes.xyxy, boxes.cls, boxes.conf):
            x1, y1, x2, y2 = map(int, box[:4])
            mask[y1:y2, x1:x2] = 0

# Apply the mask to the original image
masked_image = cv2.bitwise_and(img, mask)

fx = 15
fy = 15
global_enlarged_image = cv2.resize(masked_image, (0, 0), fx=fx, fy=fy)
# Perform global OCR on the masked image
global_ocr_result = reader.readtext(global_enlarged_image, rotation_info=[0, 90, 270])

radius= 1
offset = 0
connection_dict = {}

for i, (_, endpoints) in enumerate(contour_endpoints):
    for endpoint in endpoints:
        connection_dict[i]=[]



#loop the YOLO detection results for post processing
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
            box_boundaries = (x1, y1, x2, y2)
            enlarged_box_boundaries = (x1-offset, y1-offset, x2+offset, y2+offset)
            # Expanding the bounding box by the threshold
            expanded_x1 = max(0, x1 - expand_threshold)
            expanded_y1 = max(0, y1 - expand_threshold)
            expanded_x2 = min(original_image.shape[1], x2 + expand_threshold)
            expanded_y2 = min(original_image.shape[0], y2 + expand_threshold)
            
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            connections = []
            color = random_color()
            
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
                            "shape": class_names[int(other_class_id)],  # Assuming class names are same as shapes
                            "score": score.item() , # Convert the tensor score to a Python float
                            "bounding_box":(int(x1_other),int(y1_other),int(x2_other),int(y2_other))
                        })            
            
            
            #Post processing the connection detection
            #if len(contained_elements)==0:
                
            
            endpoint_list =[]
            contour_endpoints = sorted(contour_endpoints, key=lambda x: x[1][0] if x[1] else (float('inf'), float('inf')))
            for i, (_, endpoints) in enumerate(contour_endpoints):
                for endpoint in endpoints:
                    # Enlarge the point by a certain radius
                    enlarged_point = ((np.array(endpoint) - radius).tolist(), 
                                      (np.array(endpoint) + radius).tolist())
                    # Check if any part of the enlarged point is within the box
                    
                    if any(point_in_box(point, enlarged_box_boundaries) for point in enlarged_point):
                        
                        if i  in endpoint_list:
                            continue
                        endpoint_list.append(i)
                        try:
                            connection_dict[i].append([box_id,str(len(connections)+1),box_boundaries,endpoint])
                        except Exception as e:
                            continue
                        
                        connections.append(str(len(connections)+1))  # Add the index of the contour endpoint
                        cv2.circle(original_image, tuple(endpoint), 5, color, -1)
                        #cv2.putText(img, str(i), tuple(endpoint), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                    #else:
                        #cv2.circle(original_image, tuple(endpoint), 5, (0,0,0), -1)
                            
            cv2.rectangle(original_image, (x1-offset, y1-offset), (x2+offset, y2+offset), color, 2)            
                
                



            # Write the ID and recognized name on the original image

            element = {
                "ID": box_id,
                "shape": class_name,
                "score": score.item(),  # Convert the tensor score to a Python float
                "connections": list(set(connections)),    
                "bounding_box":box_boundaries,
                "ContainedElement": contained_elements
            }
            output["Elements"].append(element)
            
            
            cv2.putText(original_image, f"{box_id}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Display the image with IDs and names
    #cv2.imshow('Image with IDs and Names', original_image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

# Write output to JSON file

connection_list = []

for conn in connection_dict:
    if len(connection_dict[conn])==2:
        current_dict = {"from":{
            "elementGUID":connection_dict[conn][0][0],
            "elementConnectionPoint":connection_dict[conn][0][1]
            },
            "to":{
                "elementGUID":connection_dict[conn][1][0],
                "elementConnectionPoint":connection_dict[conn][1][1]
                }
            
            }
        connection_list.append(current_dict)
      
        # 3 connection points
    elif  len(connection_dict[conn])==3:
        conn_sorted = sorted(connection_dict[conn], key=lambda x: (x[-1][1], x[-1][0]))
        current_dict = {"from":{
            "elementGUID":conn_sorted[0][0],
            "elementConnectionPoint":conn_sorted[0][1]
            },
            "to":{
                "elementGUID":conn_sorted[1][0],
                "elementConnectionPoint":conn_sorted[1][1]
                }
            
            }
        connection_list.append(current_dict)
        
        #2nd
        current_dict = {"from":{
            "elementGUID":conn_sorted[1][0],
            "elementConnectionPoint":conn_sorted[1][1]
            },
            "to":{
                "elementGUID":conn_sorted[2][0],
                "elementConnectionPoint":conn_sorted[2][1]
                }
            
            }
        connection_list.append(current_dict)        
        
        
     # 6 connection points  
    elif len(connection_dict[conn])==6:
        conn_sorted = sorted(connection_dict[conn], key=lambda x: (x[-1][1], x[-1][0]))
        #define extra points
        #break
        if (abs(conn_sorted[0][-1][0]-conn_sorted[1][-1][0])<abs(conn_sorted[-1][-1][0]-conn_sorted[-2][-1][0])):
            first_pair = sorted(conn_sorted[2:4], key=lambda x: (x[-1][0]))
            second_pair = sorted(conn_sorted[4:6], key=lambda x: (x[-1][0]))
            extra_pair  = sorted(conn_sorted[0:2], key=lambda x: (x[-1][0]))

            #extra 1
            current_dict = {"from":{
                "elementGUID":extra_pair[0][0],
                "elementConnectionPoint":first_pair[0][1]
                },
                "to":{
                    "elementGUID":first_pair[1][0],
                    "elementConnectionPoint":first_pair[1][1]
                    }
                
                }
            connection_list.append(current_dict) 
            
            
            #extra 2
            current_dict = {"from":{
                "elementGUID":extra_pair[1][0],
                "elementConnectionPoint":first_pair[0][1]
                },
                "to":{
                    "elementGUID":second_pair[1][0],
                    "elementConnectionPoint":second_pair[1][1]
                    }
                
                }
            connection_list.append(current_dict)    
            
        else:
            first_pair = sorted(conn_sorted[0:2], key=lambda x: (x[-1][0]))
            second_pair = sorted(conn_sorted[2:4], key=lambda x: (x[-1][0]))
            extra_pair  = sorted(conn_sorted[4:6], key=lambda x: (x[-1][1], x[-1][0]), reverse=True)

            #extra 1
            current_dict = {"from":{
                "elementGUID":extra_pair[0][0],
                "elementConnectionPoint":first_pair[0][1]
                },
                "to":{
                    "elementGUID":first_pair[1][0],
                    "elementConnectionPoint":first_pair[1][1]
                    }
                
                }
            connection_list.append(current_dict) 
            
            
            #extra 2
            current_dict = {"from":{
                "elementGUID":extra_pair[1][0],
                "elementConnectionPoint":first_pair[0][1]
                },
                "to":{
                    "elementGUID":second_pair[1][0],
                    "elementConnectionPoint":second_pair[1][1]
                    }
                
                }
            connection_list.append(current_dict)                
        
            
        #1st pair
        current_dict = {"from":{
            "elementGUID":first_pair[0][0],
            "elementConnectionPoint":first_pair[0][1]
            },
            "to":{
                "elementGUID":first_pair[1][0],
                "elementConnectionPoint":first_pair[1][1]
                }
            
            }
        connection_list.append(current_dict)            
        
        #2nd pair
        current_dict = {"from":{
            "elementGUID":second_pair[0][0],
            "elementConnectionPoint":first_pair[0][1]
            },
            "to":{
                "elementGUID":second_pair[1][0],
                "elementConnectionPoint":second_pair[1][1]
                }
            
            }
        connection_list.append(current_dict)      
            
        
            
        
        
output["Connections"] = connection_list
output["image_path"] = img_path


json_string = json.dumps(output, indent=4)
with open("shape+boundingbox.json", "w") as json_file:
    json_file.write(json_string)
    
cv2.imwrite("shape+boundingbox.jpg",original_image)

print(f"Output saved to detected_objects.json")