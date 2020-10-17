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
import pickle

 


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

PARKING_IMG = os.path.join(IMAGE_DIR,"examplefornn.jpg")

dataname="list123"


def generateparkboxes(PARKING_IMG,dataname):
    #lets generate parking places
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
    with open('{}.data'.format(dataname), 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(car_boxes, filehandle)
        
'''


print("Cars found in frame of video:")

# Draw each box on the frame
for box in car_boxes1:
    print("Car: ", box)

    y1, x1, y2, x2 = box

    # Draw the box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

print('all boxes')

cvshow(frame)
'''

generateparkboxes(PARKING_IMG,dataname)                       

def detectparking():
    DETECT_IMG = os.path.join(IMAGE_DIR,"examplefornnP.jpg")

    with open('list123.data', 'rb') as filehandle:
        # read the data as binary data stream
        parked_car_boxes = pickle.load(filehandle)

    #lets generate parking places
    frame = cv2.imread(DETECT_IMG)

    # Convert the image from BGR color (which OpenCV uses) to RGB color
    rgb_image = frame[:, :, ::-1]

    # Run the image through the Mask R-CNN model to get results.
    results = model.detect([rgb_image], verbose=0)

    # Mask R-CNN assumes we are running detection on multiple images.
    # We only passed in one image to detect, so only grab the first result.
    r = results[0]

    # Filter the results to only grab the car / truck bounding boxes
    car_boxes = get_car_boxes(r['rois'], r['class_ids'])
    # See how much those cars overlap with the known parking spaces
    overlaps = mrcnn.utils.compute_overlaps(parked_car_boxes, car_boxes)

    # Assume no spaces are free until we find one that is free
    free_space = False

    # Loop through each known parking space box
    for parking_area, overlap_areas in zip(parked_car_boxes, overlaps):

        # For this parking space, find the max amount it was covered by any
        # car that was detected in our image (doesn't really matter which car)
        max_IoU_overlap = np.max(overlap_areas)

        # Get the top-left and bottom-right coordinates of the parking area
        y1, x1, y2, x2 = parking_area

        # Check if the parking space is occupied by seeing if any car overlaps
        # it by more than 0.15 using IoU
        if max_IoU_overlap < 0.15:
            # Parking space not occupied! Draw a green box around it
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            # Flag that we have seen at least one open space
            free_space = True
        else:
            # Parking space is still occupied - draw a red box around it
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)

            # Write the IoU measurement inside the box
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, f"{max_IoU_overlap:0.2}", (x1 + 6, y2 - 6), font, 0.3, (255, 255, 255))



        # If a space has been free for several frames, we are pretty sure it is really free!
        if free_space:
            # Write SPACE AVAILABLE!! at the top of the screen
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, f"Parking:", (10, 100), font, 1.0, (0, 255, 0), 2, cv2.FILLED)

    #cvshow(frame)
    READY_IMG = os.path.join(IMAGE_DIR,"ready.jpg")
    cv2.imwrite(READY_IMG,frame)

detectparking()

