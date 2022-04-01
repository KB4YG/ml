######## Tensorflow Imaage Classifier #########
#
# Author: Erik Handeland Date: 12/12/2021
# Description: This is a benchmark for the Tensorflow Image Classifier,
# used to test the performance of each model.
#

# Import packages
import os
from os.path import isfile, join
from img_classifier import imgClassify

def benchmark():

    CWD_PATH = os.getcwd()
    images = [f for f in os.listdir(CWD_PATH+"/test_img") if isfile(join(CWD_PATH+"/test_img", f)) and not f.startswith('.')]
    models = [f for f in os.listdir(CWD_PATH + "/models") if not f.startswith('.')]

    for img in images:
        print(img)
        for m in models:
            print("\t" + m)
            result = imgClassify("models/"+m, "test_img/"+img, DEBUG=True)
            print("\tNumber of vehicles: ", result["vehicles"])
            print("\tNumber of pedestrians: ", result["pedestrians"])
            print("\tNumber of objects: ", result["objects"])
            print()


if __name__ == "__main__":
     benchmark()