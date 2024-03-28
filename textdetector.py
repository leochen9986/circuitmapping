import argparse
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image
import json
import cv2


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



#Process the Element with OCR
for item in data["Elements"]:
    
    ocr_result = ocr.ocr(img[item["bounding_box"][1]:item["bounding_box"][3],item["bounding_box"][0]:item["bounding_box"][2]], cls=True)
    
    if ocr_result[0]:
        x1, y1 = int(ocr_result[0][0][0][0][0]) , int(ocr_result[0][0][0][0][1])
        x2, y2 = int(ocr_result[0][0][0][2][0]) , int(ocr_result[0][0][0][2][1])
         
        cv2.rectangle(draw_img[item["bounding_box"][1]:item["bounding_box"][3],item["bounding_box"][0]:item["bounding_box"][2]] , (x1, y1), (x2, y2), (0,255,0), 2)  
        item["name"] = ocr_result[0][0][1][0]
        
        cv2.putText(draw_img, str(item["name"]), (item["bounding_box"][0],item["bounding_box"][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    else:
        item["name"] = ""
        
    #Inner Loop     
    if len(item["ContainedElement"])>0:
        for contain_item in item["ContainedElement"]:
            if ocr_result[0]:
                x1, y1 = int(ocr_result[0][0][0][0][0]) , int(ocr_result[0][0][0][0][1])
                x2, y2 = int(ocr_result[0][0][0][2][0]) , int(ocr_result[0][0][0][2][1])
                 
                cv2.rectangle(draw_img[contain_item["bounding_box"][1]:contain_item["bounding_box"][3],contain_item["bounding_box"][0]:contain_item["bounding_box"][2]] , (x1, y1), (x2, y2), (0,255,0), 2)  
                contain_item["name"] = ocr_result[0][0][1][0]
                
                cv2.putText(draw_img, str(item["name"]), (contain_item["bounding_box"][0],contain_item["bounding_box"][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            else:
                contain_item["name"] = ""            
    
    

json_string = json.dumps(data, indent=4)
with open("shape+boundingbox+text.json", "w") as json_file:
    json_file.write(json_string)
    
cv2.imwrite("shape+boundingbox+text.jpg",draw_img)


