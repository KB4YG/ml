#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 20:58:55 2018

@author: YumingWu
"""

import os
import tensorflow as tf
import numpy as np
import cv2
import xml.etree.ElementTree as ET
import xmltodict
import glob


def evaluate_tflite_google(model):
    # Load the TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=model)
    interpreter.allocate_tensors()

    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test the model on random input data.
    input_shape = input_details[0]['shape']
    input_data = np.array(np.random.random_sample(input_shape), dtype=np.uint8)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(output_data)


def getObjects(xmlFile):
    xmlFile = xmlFile[:-3] + "xml"

    with open(xmlFile) as file:
        data = file.read()
        xml = xmltodict.parse(data)
        boxes = []
        objects = xml['annotation']["object"]
        if not isinstance(objects, list):
            objects = [objects]

        for obj in objects:
            # name = obj['name']
            box = obj['bndbox']
            xmin = int(box['xmin'])
            ymin = int(box['ymin'])
            xmax = int(box['xmax'])
            ymax = int(box['ymax'])

            tr = (xmax, ymax)  # Top right
            bl = (xmin, ymin)  # Bottom left

            boxes.append([bl, tr])

    return boxes


# returns true if two boxes overlap
def doBoxesOverlap(box_1, box_2):
    if (box_1[0] >= box_2[2]) or (box_1[2] <= box_2[0]) or (box_1[3] <= box_2[1]) or (box_1[1] >= box_2[3]):
        return False
    else:
        return True


def inArea(box_1, box_2, percentage):
    # Compute the area of the intersection, which is a rectangle too:
    dx = max(0, min(box_1[1][0], box_2[1][0]) - max(box_1[0][0], box_2[0][0]))
    dy = max(0, min(box_1[1][1], box_2[1][1]) - max(box_1[0][1], box_1[0][1]))

    intersecting_area = dx * dy
    b1_area = (box_1[1][0] - box_1[0][0]) * (box_1[1][1] - box_1[0][1])
    b2_area = (box_2[1][0] - box_2[0][0]) * (box_2[1][1] - box_2[0][1])

    overlap = intersecting_area / (b1_area + b2_area - intersecting_area)

    if overlap >= percentage:
        print("\t %5.2f%", overlap)
        return True
    else:
        return False


def evaluate_tflite(model, img_dir, min_conf_threshold):
    # Load the TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=model)
    interpreter.allocate_tensors()

    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    # check model type
    floating_model = (input_details[0]['dtype'] == np.float32)

    input_mean = 127.5
    input_std = 127.5

    # test_data the model on each image
    images = glob.glob(img_dir + "/*.jpg")
    for file in images:
        print(file)
        # Load image and resize to expected shape [1xHxWx3]
        image = cv2.imread(file)
        if image is None:
            print("Image not found, check path")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        imH, imW, _ = image.shape
        image_resized = cv2.resize(image_rgb, (width, height))
        input_data = np.expand_dims(image_resized, axis=0)

        # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # Retrieve detection results
        boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates of detected objects
        scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence of detected objects

        expected = getObjects(file)

        # Loop over all detections and draw detection box if confidence is above minimum threshold
        results = []
        for i in range(len(scores)):
            if (scores[i] > min_conf_threshold) and (scores[i] <= 1.0):
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
                results.append([bl, tr])

        count = 0
        for res in results:
            for box in expected:
                if inArea(res, box, 0.01): #TODO optmi for label .. only cars
                    count += 1
                    break

        print("found:", count)
        print("expected:", len(expected))

