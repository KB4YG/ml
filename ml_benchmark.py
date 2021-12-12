######## Tensorflow Imaage Classifier #########
#
# Author: Erik Handeland Date: 12/12/2021
# Description: This is a benchmark for the Tensorflow Image Classifier,
# used to test the performance of each model.
#

# TODO: get test data that reflects project area and hardcode img paths with expect values
# automate selection of optimal algorithm take into account vehicle/person/animal classification

# Import packages
import os
from os.path import isfile, join
from img_classifier import imgClassify

def benchmark():
    CWD_PATH = os.getcwd()
    images = [f for f in os.listdir(CWD_PATH+"/test_img") if isfile(join(CWD_PATH+"/test_img", f))]
    images.pop(0)  # remove the .DS_Store file
    models = [f for f in os.listdir(CWD_PATH + "/models")]
    models.pop(0)  # remove the .DS_Store file

    for img in images:
        print(img)
        for m in models:
            objects, count = imgClassify("models/"+m, "test_img/"+img)
            print(m)
            print(count)
            print(objects)
            print()


if __name__ == "__main__":
     benchmark()