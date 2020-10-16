import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

import cv2
import mrcnn.config
import mrcnn.utils
from mrcnn.model import MaskRCNN


# Configuration that will be used by the Mask-RCNN library
class MaskRCNNConfig(mrcnn.config.Config):
    NAME = "coco_pretrained_model_config"
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    NUM_CLASSES = 1 + 80  # COCO dataset has 80 classes + one background class
    DETECTION_MIN_CONFIDENCE = 0.6


# Filter a list of Mask R-CNN detection results to get only the detected cars / trucks
def get_car_boxes(boxes, class_ids):
    car_boxes = []

    for i, box in enumerate(boxes):
        # If the detected object isn't a car / truck, skip it
        if class_ids[i] in [3, 8, 6]:
            car_boxes.append(box)

    return np.array(car_boxes)


#for testing - opens image in openCV

def cvshow(img):
    cv2.imshow('img', img)
    cv2.waitKey(0)  




# Root directory of the project
ROOT_DIR = Path(".")

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Create a Mask-RCNN model in inference mode
model = MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=MaskRCNNConfig())

# Load pre-trained model
model.load_weights(COCO_MODEL_PATH, by_name=True)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

PARKING_IMG = os.path.join(IMAGE_DIR,"parking.jpg")

frame = cv2.imread(PARKING_IMG)

# Convert the image from BGR color (which OpenCV uses) to RGB color
rgb_image = frame[:, :, ::-1]

# Run the image through the Mask R-CNN model to get results.
results = model.detect([rgb_image], verbose=0)

# Mask R-CNN assumes we are running detection on multiple images.
# We only passed in one image to detect, so only grab the first result.
r = results[0]

# The r variable will now have the results of detection:
# - r['rois'] are the bounding box of each detected object
# - r['class_ids'] are the class id (type) of each detected object
# - r['scores'] are the confidence scores for each detection
# - r['masks'] are the object masks for each detected object (which gives you the object outline)

# Filter the results to only grab the car / truck bounding boxes
car_boxes = get_car_boxes(r['rois'], r['class_ids'])

print("Cars found in frame of video:")

# Draw each box on the frame
for box in car_boxes:
    print("Car: ", box)

    y1, x1, y2, x2 = box

    # Draw the box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

print('all boxes')

cvshow(frame)

READY_IMG = os.path.join(IMAGE_DIR,"ready.jpg")
cv2.imwrite(READY_IMG,frame)
                         
print('finish')
