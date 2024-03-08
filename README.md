# AI Circuit Mapping

## Prerequisite 
- Python  >= 3.8 
- CUDA 11 (to utilize GPU)
## Run project
1) Install dependencies

```
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics
pip install easyocr
```

3) Run training
The dataset are in the folders train, test , val
```
python train.py
```

3) Run test prediction (Object Detection)

```
python predict.py
```

4) Run line detection (Milestone2)

```
python line_detection.py path_to_your_image.jpg
```