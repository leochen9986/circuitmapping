from ultralytics import YOLO
import os
import shutil

# Load a model
model = YOLO('best.pt')  # pretrained YOLOv8n model

# Define input and output directories
input_dir = r'C:\Users\winte\OneDrive\upwork\Ayesh-circuit\Ayeshdata\valid\images'
output_dir = 'output'

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# List all image files in the input directory
image_files = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

# Run batched inference on the list of images
results = model([os.path.join(input_dir, img) for img in image_files])

# Process results list and save the output images
for img, result in zip(image_files, results):
    result.save(filename=os.path.join(output_dir, img))  # save to output directory
