######## Tensorflow Imaage Classifier #########
#
# Author: Erik Handeland Date: 12/12/2021
# Description: This is a benchmark for the Tensorflow Image Classifier,
# used to test the performance of each model-metadata.
#

# Import packages
import os
import csv
from os.path import isfile, join
from obj_detection import objDetection


def benchmark():
    CWD_PATH = os.getcwd()
    f = open(CWD_PATH + "/tests/benchmark/results.csv", 'w')
    writer = csv.writer(f)
    images = [f for f in os.listdir(CWD_PATH + "/images") if
              isfile(join(CWD_PATH + "/images", f)) and not f.startswith('.')]
    models = [f for f in os.listdir(CWD_PATH + "/models") if not f.startswith('.')]
    writer.writerow(["image", "type"] + models)
    errors = []

    for img in images:
        print(img)
        vehicles = []
        pedestrians = []
        objects = []
        print(f'starting img {img}')
        for m in models:
            print(f'starting Model {m}')
            result = objDetection("models/" + m, "images/" + img)
            vehicles.append(result["vehicles"])
            pedestrians.append(result["pedestrians"])
            objects.append(result["objects"])
            if result["error"]:
                errors.append((m, result["error"]))
                models.remove(m)
        print(f'Img {img} done')
        writer.writerow([img, "vehicles"] + vehicles)
        writer.writerow([img, "pedestrians"] + pedestrians)
        writer.writerow([img, "objects"] + objects)
        writer.writerow([])

    writer.writerow(["", "errors"] + errors)


if __name__ == "__main__":
    benchmark()
