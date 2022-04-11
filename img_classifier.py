######## Tensorflow Imaage Classifier #########
#
# Author: Erik Handeland Date: 12/12/2021
# Description: This program uses a TensorFlow Lite object detection model to
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


# Extract metadata from the .tflite file
def load_metadata_labels(PATH_TO_MODEL):
    label_list = []

    try:
        displayer = metadata.MetadataDisplayer.with_model_file(PATH_TO_MODEL)
        file_name = displayer.get_packed_associated_file_list()[0]
    except ValueError:
        # The model does not have metadata.
        return label_list

    if file_name:
        label_map_file = displayer.get_associated_file_buffer(file_name).decode()
        label_list = list(filter(len, label_map_file.splitlines()))
    return label_list


def imgClassify(MODEL_PATH: str, IMG_PATH, min_conf_threshold=0.50,
                GRAPH_NAME="detect.tflite", LABELMAP_NAME="labelmap.txt", BENCHMARK=False, COORDS=False):
    objects = []

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
    # Path to .tflite file, which contains the model that is used for object detection
    PATH_TO_MODEL = os.path.join(CWD_PATH, MODEL_PATH, GRAPH_NAME)
    if not exists(PATH_TO_MODEL):
        print("detect.tflite not found! at path: " + PATH_TO_MODEL)
        return {
            "error": "Invalid model path",
            "vehicles": -1,
            "pedestrians": -1,
            "confidence-threshold": min_conf_threshold,
            "objects": objects,
        }

    # Load label list from metadata or from labelmap file
    labels = load_metadata_labels(PATH_TO_MODEL)

    if not labels:  # DEPRECATED this is the old way of loading labels, new ML models should have it as metadata
        PATH_TO_LABELS = os.path.join(CWD_PATH, MODEL_PATH, LABELMAP_NAME)
        if not exists(PATH_TO_LABELS):
            print("No labelmap in metadata and no labelmap.txt found! at path: " + PATH_TO_LABELS)
            return {
                "error": "No labelmap found",
                "vehicles": -1,
                "pedestrians": -1,
                "confidence-threshold": min_conf_threshold,
                "objects": objects,
            }
        # Load the label map
        with open(PATH_TO_LABELS, 'r') as f:
            labels = [line.strip() for line in f.readlines()]

    # Load the Tensorflow Lite model.
    interpreter = Interpreter(model_path=PATH_TO_MODEL)
    interpreter.allocate_tensors()

    # Get model details
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
        print("Image not found, check path")
        return {
            "error": "Image not found, check path",
            "vehicles": -1,
            "pedestrians": -1,
            "confidence-threshold": min_conf_threshold,
            "objects": objects,
        }
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
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence of detected objects

    # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores)):

        if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
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

    if BENCHMARK:
        IMG_PATH = os.path.join(CWD_PATH + "/benchmark/" + MODEL_PATH, IMG_PATH[:-4] + "_box.jpg")
        cv2.imwrite(IMG_PATH, image)

    return {
        "error": "",
        "vehicles": cars,
        "pedestrians": people,
        "confidence-threshold": min_conf_threshold,
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
