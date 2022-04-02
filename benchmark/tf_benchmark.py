######## Tensorflow Imaage Classifier #########
#
# Author: Erik Handeland Date: 12/12/2021
# Description: Benchmarks the performance of the Tensorflow models
#
# Import packages
from tf_evaluation import evaluate_tflite

import os
def benchmark():
    BASE_DIR = os.path.join(os.path.dirname(__file__), '..')

    models = [f for f in os.listdir(BASE_DIR + "/models") if not f.startswith('.')]

    for m in models:
        evaluate_tflite(BASE_DIR + "/models/" + m + "/detect.tflite", "/home/erik/PycharmProjects/ml/test_data", 0.50)



if __name__ == "__main__":
    benchmark()