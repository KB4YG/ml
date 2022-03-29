######## Tensorflow Imaage Classifier #########
#
# Author: Erik Handeland Date: 12/12/2021
# Description: This program uses a TensorFlow Lite object detection model to
# perform object detection on an image. It creates a json file containing a
# list of detected objects and the count for each object. It also save a copy
# of the image with draws boxes and scores around the objects of interest in each image.
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
import cv2
import numpy as np
import importlib.util


def imgClassify(MODEL_NAME: str, IM_NAME='test1.jpg', min_conf_threshold=0.50,
                GRAPH_NAME="detect.tflite", LABELMAP_NAME="labelmap.txt", DEBUG=False):
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

    # Load image and resize to expected shape [1xHxWx3]
    image = cv2.imread(IM_NAME)
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
    objects = []
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence of detected objects
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

            # Corners of the bounding box
            tr = (xmax, ymax)  # Top right
            bl = (xmin, ymin)  # Bottom left
            br = (xmax, ymin)
            tl = (xmin, ymax)

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

            # Create data object and append to list of detected objects
            obj = {
                "name": object_name,
                "confidence": scores[i],
                "coord": {"top-left": tl, "top-right": tr, "bottom-right": br, "bottom-left": bl}
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

    result = {
        "vehicles": cars,
        "pedestrians": people,
        "confidence-threshold": min_conf_threshold,
        "objects": objects,
    }

    if DEBUG:
        print("cars: ", cars)
        print("people: ", people)

        IMG_PATH = os.path.join(CWD_PATH + "/benchmark/" + MODEL_NAME, IM_NAME[:-4] + "_box.png")
        cv2.imwrite(IMG_PATH, image)
    return result


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
