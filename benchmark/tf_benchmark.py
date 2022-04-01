######## Tensorflow Imaage Classifier #########
#
# Author: Erik Handeland Date: 12/12/2021
# Description: Benchmarks the performance of the Tensorflow models
#
# Import packages
from os.path import isfile, join

from tf_evaluation import evaluate_tflite

model_path = '../models/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29'
dataset_path = 'Prediction_set/'
# evaluate_tflite(model_path + '/detect.tflite', "/home/erik/PycharmProjects/ml/test_img") .. kinda works, basiclly just calls imag_classifier migh tneed to use in area for testing?

import numpy as np
import os
# !pip install -q tflite-model-maker
# !pip install -q tflite-support
from tflite_model_maker.config import ExportFormat, QuantizationConfig
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector

from tflite_support import metadata

import tensorflow as tf

assert tf.__version__.startswith('2')

tf.get_logger().setLevel('ERROR')
from absl import logging

logging.set_verbosity(logging.ERROR)


def benchmark():
    BASE_DIR = os.path.join(os.path.dirname(__file__), '..')

    models = [f for f in os.listdir(BASE_DIR + "/models") if not f.startswith('.')]

    test_data = object_detector.DataLoader.from_pascal_voc(
        BASE_DIR + '/test_data',
        BASE_DIR + '/test_data',
        ['car', 'motorcycle', 'person', 'bus', 'bicycle']
    )

    models = ["kb4yg_v1.tflite"]
    for m in models:
        print("\t" + m)
        # model.evaluate_tflite('kb4yg_v1.tflite', test_data)

        spec = model_spec.get('efficientdet_lite0')
        model = object_detector.create(test_data, model_spec=spec, do_train=False)

        # only seems to work on models that have the labelmap inside the tflite metadata, so can't run on models with
        # labelmap
        result = model.evaluate_tflite("/home/erik/PycharmProjects/ml/models/" + m, test_data)
        print(result)


if __name__ == "__main__":
    benchmark()
