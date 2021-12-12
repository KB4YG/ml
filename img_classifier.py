######## Tensorflow Imaage Classifier #########
#
# Author: Erik Handeland Date: 12/12/2021 Description: This program uses a
# TensorFlow Lite object detection model to perform object detection on an
# image. It creates a json file containing a list of detected objects and
# the count for each object. It also save a copy of the image with draws
# boxes and scores around the objects of interest in each image.
#
# This code is based off the TensorFlow Lite image classification example at:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py
# Add the following github repo by Evan Juras:
# https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi
#
# Removed unnecessary features and customised data formatting for exporting
# to external services

# Import packages
import os
import argparse
import cv2
import numpy as np
import sys
import glob
import importlib.util
from datetime import datetime
from collections import OrderedDict
from collections import defaultdict
import json

now = datetime.now()

# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    required=True)
parser.add_argument('--graph',
                    help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels',
                    help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold',
                    help='Minimum confidence threshold for displaying detected objects',
                    default=0.5)
parser.add_argument('--image',
                    help='Name of the single image to perform detection on. To run detection on multiple images, use --imagedir',
                    default=None)
parser.add_argument('--imagedir',
                    help='Name of the folder containing images to perform detection on. Folder must contain only images.',
                    default=None)

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)

# Parse input image name and directory.
IM_NAME = args.image
IM_DIR = args.imagedir

# If both an image AND a folder are specified, throw an error
if (IM_NAME and IM_DIR):
    print(
        'Error! Please only use the --image argument or the --imagedir argument, not both. Issue "python TFLite_detection_image.py -h" for help.')
    sys.exit()

# If neither an image or a folder are specified, default to using 'test1.jpg' for image name
if (not IM_NAME and not IM_DIR):
    IM_NAME = 'test1.jpg'

# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
else:
    from tensorflow.lite.python.interpreter import Interpreter

# Get path to current working directory
CWD_PATH = os.getcwd()

# Define path to images and grab all image filenames
if IM_DIR:
    PATH_TO_IMAGES = os.path.join(CWD_PATH, IM_DIR)
    images = glob.glob(PATH_TO_IMAGES + '/*')

elif IM_NAME:
    PATH_TO_IMAGES = os.path.join(CWD_PATH, IM_NAME)
    images = glob.glob(PATH_TO_IMAGES)

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH, MODEL_NAME, LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labels[0] == '???':
    del (labels[0])

# Load the Tensorflow Lite model.
interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Loop over every image and perform detection
for image_path in images:

    # Load image and resize to expected shape [1xHxWx3]
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    imH, imW, _ = image.shape
    image_resized = cv2.resize(image_rgb, (width, height))
    input_data = np.expand_dims(image_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Retrieve detection results
    objects = defaultdict(int)
    boxes = interpreter.get_tensor(output_details[0]['index'])[
        0]  # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[
        0]  # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[
        0]  # Confidence of detected objects
    num = 0  # Total number of detected objects

    # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores)):
        if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
            num += 1

            # Get bounding box coordinates and draw box Interpreter can
            # return coordinates that are outside of image dimensions,
            # need to force them to be within image using max() and min()
            ymin = int(max(1, (boxes[i][0] * imH)))
            xmin = int(max(1, (boxes[i][1] * imW)))
            ymax = int(min(imH, (boxes[i][2] * imH)))
            xmax = int(min(imW, (boxes[i][3] * imW)))

            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)

            # Draw label
            object_name = labels[int(classes[
                                         i])]  # Look up object name from "labels" array using class index
            object_score = int(scores[i] * 100)
            label = '%s: %d%%' % (
                object_name, object_score)  # Example: 'person: 72%'
            labelSize, baseLine = cv2.getTextSize(label,
                                                  cv2.FONT_HERSHEY_SIMPLEX,
                                                  0.7, 2)  # Get font size
            label_ymin = max(ymin, labelSize[
                1] + 10)  # Make sure not to draw label too close to top of window
            cv2.rectangle(image, (xmin, label_ymin - labelSize[1] - 10),
                          (xmin + labelSize[0], label_ymin + baseLine - 10),
                          (255, 255, 255),
                          cv2.FILLED)  # Draw white box to put label text in
            cv2.putText(image, label, (xmin, label_ymin - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0),
                        2)  # Draw label text
            objects[object_name] += 1  # Increment counter hashmap counter

    # Save image or display image viewer for img dir
    if IM_NAME:
        objects = OrderedDict(sorted(objects.items(), key=lambda x: x[1],
                                     reverse=True))  # Sort hashmap by value
        for k, v in objects.items():
            print(k, v)
        json_object = json.dumps(objects, indent=4)
        date = now.strftime("%m-%d-%Y %H:%M")
        IMG_PATH = os.path.join(CWD_PATH, "img", date + ".png")
        cv2.imwrite(IMG_PATH, image)

    else:
        cv2.imshow('Object detector', image)

        # Press any key to continue to next image, or press 'q' to quit
        if cv2.waitKey(0) == ord('q'):
            break

# Clean up
cv2.destroyAllWindows()
