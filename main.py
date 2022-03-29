# Just a wrapper to call img_classifier from the command line
# Run: python main.py --location "Fitton Green" --image "image.jpg"

from img_classifier import imgClassify
import argparse


def run():
    result = imgClassify(MDL_PATH, IMG_PATH)
    print("Number of vehicles: ", result["vehicles"])
    print("Number of pedestrians: ", result["pedestrians"])
    print("Number of objects: ", result["objects"])


# get commandline arguments
parser = argparse.ArgumentParser()
parser.add_argument('--image',
                    help='image to classify',
                    type=str,
                    default='test_img/parking-lot-8.jpg')
parser.add_argument('--model',
                    help='Path to model folder',
                    type=str,
                    default='models/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29')

args = parser.parse_args()
MDL_PATH = args.model
IMG_PATH = args.image

if __name__ == '__main__':
    run()
