######## Tensorflow Imaage Classifier #########
#
# Author: Erik Handeland Date: 12/12/2021
# Description: This program uses a TensorFlow Lite object detection model-metadata to
# perform object detection on an image. It creates a dict containing a
# list of detected objects and the count for each object. It also save a copy
# of the image with draws boxes and scores around the objects of interest for each image.
#
# This code is based off the TensorFlow Lite image classification example at:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py
# Add the following github repo by Evan Juras:
# https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi
#

# Import packages
import os
from os.path import exists
import cv2
import numpy as np
import importlib.util
from tflite_support import metadata
from from_root import from_root

# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
else:
    from tensorflow.lite.python.interpreter import Interpreter


# Extract metadata from the .tflite file
def load_metadata_labels(PATH_TO_MODEL):
    label_list = []

    try:
        displayer = metadata.MetadataDisplayer.with_model_file(PATH_TO_MODEL)
        file_name = displayer.get_packed_associated_file_list()[0]
    except ValueError:
        # The model-metadata does not have metadata.
        return label_list

    if file_name:
        label_map_file = displayer.get_associated_file_buffer(file_name).decode()
        label_list = list(filter(len, label_map_file.splitlines()))
    return label_list


def load_labels(PATH_TO_GRAPH, PATH_TO_LABELS):
    # Load label list from metadata or from labelmap file
    label_list = load_metadata_labels(PATH_TO_GRAPH)

    if not label_list:  # DEPRECATED this is the old way of loading labels, new ML models should have it as metadata
        if not exists(PATH_TO_LABELS):
            print("No labelmap in metadata and no labelmap.txt found! at path: " + PATH_TO_LABELS)
            return {
                "error": "No labelmap found",
                "vehicles": -1,
                "pedestrians": -1,
                "confidence-threshold": 0.50,
                "objects": [],
            }
        # Load the label map
        with open(PATH_TO_LABELS, 'r') as f:
            label_list = [line.strip() for line in f.readlines()]

    return label_list


# MODEL_NAME: should be the name of a directory in the models directory
# IMG_PATH: should be the path full path to your target image
# COORDS: Whether to return coordinates of detected objects
# MIN_CONF_LEVEL: is the minimum confidence level to be considered a detection 0-1
# PATH_TO_GRAPH & LABELMAP_NAME: Name of the .tflite file and the labelmap file. Defaults should work for most cases
# SAVED_IMG_PATH: Directory to save the image with boxes and scores. If not specified, no image will be saved
def objDetection(MODEL_NAME: str, IMG_PATH: str, MIN_CONF_LEVEL=0.50,
                 GRAPH_NAME="detect.tflite", LABELMAP_NAME="labelmap.txt", SAVED_IMG_PATH="", COORDS=False):
    objects = []

    # Get path to project root
    CWD_PATH = str(from_root())
    # Path to .tflite file, which contains the model-metadata that is used for object detection
    try:  # running from pip install - pip install has different path structure that source
        PATH_TO_MODEL = os.path.join(CWD_PATH, "models", MODEL_NAME)
        PATH_TO_GRAPH = os.path.join(PATH_TO_MODEL, GRAPH_NAME)
        PATH_TO_LABELS = os.path.join(PATH_TO_MODEL, LABELMAP_NAME)
        if not exists(PATH_TO_GRAPH):
            raise FileNotFoundError
    except FileNotFoundError:  # running from source
        PATH_TO_MODEL = os.path.join(CWD_PATH, "obj_detection", "models", MODEL_NAME)
        PATH_TO_GRAPH = os.path.join(PATH_TO_MODEL, GRAPH_NAME)
        PATH_TO_LABELS = os.path.join(PATH_TO_MODEL, LABELMAP_NAME)
        if not exists(PATH_TO_GRAPH):
            print("detect.tflite not found! at path: " + PATH_TO_GRAPH)
            return {
                "error": "Invalid model-metadata path",
                "vehicles": -1,
                "pedestrians": -1,
                "confidence-threshold": MIN_CONF_LEVEL,
                "objects": objects,
            }

    # Load label list from metadata or from labelmap file
    labels = load_labels(PATH_TO_GRAPH, PATH_TO_LABELS)

    # Load the Tensorflow Lite model-metadata.
    interpreter = Interpreter(model_path=PATH_TO_GRAPH)
    interpreter.allocate_tensors()

    # Get model-metadata details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    floating_model = (input_details[0]['dtype'] == np.float32)
    input_mean = 127.5
    input_std = 127.5

    # Load image and resize to expected shape [1xHxWx3]
    image = cv2.imread(IMG_PATH)
    if image is None:
        print("Image not found, check path ", IMG_PATH)
        return {
            "error": "Image not found, check path",
            "vehicles": -1,
            "pedestrians": -1,
            "confidence-threshold": MIN_CONF_LEVEL,
            "objects": objects,
        }

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    imH, imW, _ = image.shape
    image_resized = cv2.resize(image_rgb, (width, height))
    input_data = np.expand_dims(image_resized, axis=0)

    # Normalize pixel values if using a floating model-metadata (i.e. if model-metadata is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model-metadata with the image as input
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Retrieve detection results
    try:
        boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates of detected objects
        classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index of detected objects
        scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence of detected objects
    except:
        return {
            "error": "Invalid model-metadata output details, probably using model-metadata for JS or Dart",
            "vehicles": -1,
            "pedestrians": -1,
            "confidence-threshold": MIN_CONF_LEVEL,
            "objects": objects,
        }

    # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores)):

        if ((scores[i] > MIN_CONF_LEVEL) and (scores[i] <= 1.0)):
            # Get bounding box coordinates and draw box Interpreter can
            # return coordinates that are outside of image dimensions,
            # need to force them to be within image using max() and min()
            ymin = int(max(1, (boxes[i][0] * imH)))
            xmin = int(max(1, (boxes[i][1] * imW)))
            ymax = int(min(imH, (boxes[i][2] * imH)))
            xmax = int(min(imW, (boxes[i][3] * imW)))

            # Corners of the bounding box
            tr = (xmax, ymax)  # Top right
            bl = (xmin, ymin)  # Bottom left
            br = (xmax, ymin)
            tl = (xmin, ymax)

            # Draw detection box on image
            cv2.rectangle(image, bl, tr, (10, 255, 0), 2)
            # Draw label
            object_name = labels[int(classes[i])]  # Look up object name from "labels" array using class index
            object_score = int(scores[i] * 100)
            label = '%s: %d%%' % (object_name, object_score)  # Example: 'person: 72%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)  # Get font size
            label_ymin = max(ymin, labelSize[1] + 10)  # Make sure not to draw label too close to top of window
            cv2.rectangle(image, (xmin, label_ymin - labelSize[1] - 10),
                          (xmin + labelSize[0], label_ymin + baseLine - 10),
                          (255, 255, 255), cv2.FILLED)  # Draw white box to put label text in
            cv2.putText(image, label, (xmin, label_ymin - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)  # Draw label text

            # Add object to objects list
            obj = {
                "name": object_name,
                "confidence": scores[i],
                "coord": {"top-left": tl, "top-right": tr, "bottom-right": br, "bottom-left": bl} if COORDS else {},
            }
            objects.append(obj)

    # count vehicles and pedestrians
    cars = 0
    people = 0
    for obj in objects:
        if obj["name"] == "car" or obj["name"] == "truck":
            cars += 1
        elif obj["name"] == "person":
            people += 1

    if SAVED_IMG_PATH:
        _, tail = os.path.split(IMG_PATH)
        SAVED_IMG_PATH = os.path.join(SAVED_IMG_PATH, tail[:-4] + "_box.jpg")
        cv2.imwrite(SAVED_IMG_PATH, image)

    return {
        "error": "",
        "vehicles": cars,
        "pedestrians": people,
        "confidence-threshold": MIN_CONF_LEVEL,
        "objects": objects,
    }


# Sample function for detecting if object is in a certain area, useful if some parking lots have handicapped or
# oversize parking spaces
# if inArea([tr, tl, br, bl], (100, 400), (800, 600)):
#       print("Object detected in area")
def inArea(points, box_start, box_end):
    for point in points:
        if (box_start[0] < point[0] < box_end[0] and
                box_start[1] < point[1] < box_end[1]):
            return True
    return False
