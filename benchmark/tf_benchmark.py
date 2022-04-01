######## Tensorflow Imaage Classifier #########
#
# Author: Erik Handeland Date: 12/12/2021
# Description: Benchmarks the performance of the Tensorflow models
#
# Import packages
import os
from os.path import isfile, join

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

    CWD_PATH = os.getcwd()
    images = [f for f in os.listdir(CWD_PATH+"/test_img") if isfile(join(CWD_PATH+"/test_img", f)) and not f.startswith('.')]
    models = [f for f in os.listdir(CWD_PATH + "/models") if not f.startswith('.')]

    test_data = object_detector.DataLoader.from_pascal_voc(
        'kb4yg/test',
        'kb4yg/test',
        ['car', 'motorcycle', 'person', 'bus', 'bicycle']
    )

    for img in images:
        print(img)
        for m in models:
            print("\t" + m)
            # model.evaluate_tflite('android.tflite', test_data)
            object_detector.evaluate_tflite("models/"+m, test_data)


            train_data = object_detector.DataLoader.from_pascal_voc(
                'kb4yg/train',
                'kb4yg/train',
                ['car', 'motorcycle','person','bus','bicycle']
            )
            spec = model_spec.get('efficientdet_lite0')
            model = object_detector.create(train_data, model_spec=spec, batch_size=12, train_whole_model=True,
                                           epochs=20, validation_data=train_data)
            result = model.evaluate_tflite("models/"+m, test_data)
            # result = imgClassify("models/"+m, "test_img/"+img, DEBUG=True)


if __name__ == "__main__":
     benchmark()