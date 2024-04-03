import argparse
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image
import json
import cv2

import math

def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def is_overlap_y(box1, box2):
    # Unpack the coordinates of the bounding boxes
    _, y1_1, _, y2_1 = box1
    _, y1_2, _, y2_2 = box2

    # Check if the boxes overlap on the y-axis
    return max(y1_1, y1_2) < min(y2_1, y2_2)

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--json_file_path', type=str, help='Path to the JSON file')
parser.add_argument('--img_path', type=str, help='Path to the original image')
parser.add_argument('--drawn_img_pth', type=str, help='Path to the drawn image')
args = parser.parse_args()

json_file_path = args.json_file_path

# Open the file in read mode
with open(json_file_path, 'r') as file:
    # Load the JSON data
    data = json.load(file)


# Path to the original image
img_path = args.img_path
drawn_img_pth = args.drawn_img_pth

img = cv2.imread(img_path)
draw_img = cv2.imread(drawn_img_pth)
# Load the OCR model
ocr = PaddleOCR(use_angle_cls=True, lang='en')
rotation_angles = [0, 90, 180, 270]
match_list = ["m1","m2","m3","m4",
              "rx","tx","bo","mcard",
              "connector","clock","receiver","esd","cmc",
              "type c conn","r1","r2","hdmi connector","via","pch",
              "cap","choke","post-cap","pre-cap",
              "bo'","mr'","mr","c1","c2"]
#Process the Element with OCR
for item in data["Elements"]:
    
    ocr_result = None
    #Inner Loop     
    if len(item["ContainedElement"])>0:
        for contain_item in item["ContainedElement"]:
            
            cropped_img = img[contain_item["bounding_box"][1]:contain_item["bounding_box"][3], contain_item["bounding_box"][0]:contain_item["bounding_box"][2]].copy()
            for angle in rotation_angles:
                rotated_img = cv2.rotate(cropped_img, cv2.ROTATE_90_CLOCKWISE)
                rescaled_img = cv2.resize(rotated_img, (0, 0), fx=6, fy=6)
                ocr_result = ocr.ocr(rescaled_img, cls=True)
                
                if ocr_result[0] and ocr_result[0][0][1][0] in match_list:
                    break  # Break the loop if the OCR result matches an element in your list            
                    
            
            if ocr_result[0]:
                combined_string = ""
                for outer_list in ocr_result:
                    for inner_list in outer_list:
                        for itemx in inner_list:
                            if isinstance(itemx, tuple):  # Check if the item is a tuple
                                combined_string += itemx[0]  # Concatenate the string from the tuple      
                                combined_string += " "
                
                x1, y1 = int(ocr_result[0][0][0][0][0]) , int(ocr_result[0][0][0][0][1])
                x2, y2 = int(ocr_result[0][0][0][2][0]) , int(ocr_result[0][0][0][2][1])
                 
                #cv2.rectangle(draw_img[contain_item["bounding_box"][1]:contain_item["bounding_box"][3],contain_item["bounding_box"][0]:contain_item["bounding_box"][2]] , (x1, y1), (x2, y2), (0,255,0), 2)  
                contain_item["name"] = combined_string.strip()
                
                cv2.putText(draw_img, str(contain_item["name"]), (contain_item["bounding_box"][0],contain_item["bounding_box"][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                 
            else:
                contain_item["name"] = "" 
            cv2.rectangle(img, (contain_item["bounding_box"][0], contain_item["bounding_box"][1]), (contain_item["bounding_box"][2], contain_item["bounding_box"][3]), (255, 255, 255), -1) 
                
            
            
    
    
    
    cropped_img = img[item["bounding_box"][1]:item["bounding_box"][3], item["bounding_box"][0]:item["bounding_box"][2]].copy()
    for angle in rotation_angles:
        rotated_img = cv2.rotate(cropped_img, cv2.ROTATE_90_CLOCKWISE)
        rescaled_img = cv2.resize(rotated_img, (0, 0), fx=6, fy=6)
        ocr_result = ocr.ocr(rescaled_img, cls=True)
        
        print(ocr_result)
        if ocr_result[0] and ocr_result[0][0][1][0] in match_list:
            break  # Break the loop if the OCR result matches an element in your list

    if ocr_result[0]:
        combined_string = ""
        for outer_list in ocr_result:
            for inner_list in outer_list:
                for itemx in inner_list:
                    if isinstance(itemx, tuple):  # Check if the item is a tuple
                        combined_string += itemx[0]  # Concatenate the string from the tuple      
                        combined_string += " "
        
        x1, y1 = int(ocr_result[0][0][0][0][0]) , int(ocr_result[0][0][0][0][1])
        x2, y2 = int(ocr_result[0][0][0][2][0]) , int(ocr_result[0][0][0][2][1])
         
        #cv2.rectangle(draw_img[item["bounding_box"][1]:item["bounding_box"][3],item["bounding_box"][0]:item["bounding_box"][2]] , (x1, y1), (x2, y2), (0,255,0), 2)  
        item["name"] = combined_string.strip()
        if item["shape"]=="resistor" and (item["name"].lower()=="w" or item["name"].lower()=="v" or item["name"].lower()=="ww"or item["name"].lower()=="mw"):
            item["name"] = ""
        

        cv2.putText(draw_img, str(item["name"]), (item["bounding_box"][0],item["bounding_box"][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
    else:
        item["name"] = ""
    

        
    cv2.rectangle(img, (item["bounding_box"][0], item["bounding_box"][1]), (item["bounding_box"][2], item["bounding_box"][3]), (255, 255, 255), -1) 
        

###RUN Overall OCR for those which are not detected yet
ocr_result = ocr.ocr(img, cls=True)

#Collect them into proper list of dict
final_ocr_result = []
if ocr_result[0]:  
    for ocritem in ocr_result[0]:
        x1, y1 = int(ocritem[0][0][0]) , int(ocritem[0][0][1])
        x2, y2 = int(ocritem[0][2][0]) , int(ocritem[0][2][1])
        text = ocritem[1][0]
        cv2.rectangle(draw_img, (x1,y1),(x2,y2), (0, 0, 0), 1)
        final_ocr_result.append({"text":text,"position":(x1, y1,x2, y2)})

sorted_final_ocr_result = sorted(final_ocr_result, key=lambda x: (x['position'][1], x['position'][0]))

for item in data["Elements"]:
    if item["name"] == "":
        for ocritem in  sorted_final_ocr_result:
            distance1= calculate_distance(item["bounding_box"][0], item["bounding_box"][1], ocritem['position'][0], ocritem['position'][1])
            distance2= calculate_distance(item["bounding_box"][2], item["bounding_box"][3], ocritem['position'][2], ocritem['position'][3])
            threshold_dis =100
            if ((item["bounding_box"][0] <= ocritem['position'][2] and item["bounding_box"][2] >= ocritem['position'][0]) or (is_overlap_y(tuple(item["bounding_box"]), ocritem['position'])) )  and (distance1<threshold_dis or distance2<threshold_dis) and ocritem["text"].lower() in match_list:
                item["name"]+=ocritem["text"]
                item["name"]+=" "
    item["name"] = item["name"].strip()      
    if item["name"].lower() == "via via":
        item["name"] = "via"
      
    cv2.putText(draw_img, str(item["name"]), (item["bounding_box"][0],item["bounding_box"][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

json_string = json.dumps(data, indent=4)
with open("shape+boundingbox+text.json", "w") as json_file:
    json_file.write(json_string)
    
cv2.imwrite("shape+boundingbox+text.jpg",draw_img)