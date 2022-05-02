import numpy as np
import os

from tflite_model_maker.config import QuantizationConfig
from tflite_model_maker.config import ExportFormat
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector

import tensorflow as tf

assert tf.__version__.startswith('2')

tf.get_logger().setLevel('ERROR')
from absl import logging

logging.set_verbosity(logging.ERROR)

spec = model_spec.get('efficientdet_lite0')
# dataset
imgs = "img2"
annot = "annot"

#train_data, validation_data, test_data = object_detector.DataLoader.from_csv('gs://cloud-ml-data/img/openimage/csv/salads_ml_use.csv')
train_data = object_detector.DataLoader.from_pascal_voc(imgs, annot, label_map={1: "apple"})

#
model = object_detector.create(train_data, model_spec=spec, batch_size=8, train_whole_model=True, validation_data=train_data)

print(model.evaluate(train_data))

print(model.evaluate_tflite(train_data))

model.export(export_dir='.')

model.evaluate_tflite('model.tflite', train_data)