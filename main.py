# Just a wrapper to call img_classifier from the command line
# Run: python main.py --image "image.jpg"

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
                    default='test_img/parking-lot-5.jpg')
parser.add_argument('--model',
                    help='Path to model folder',
                    type=str,
                    default='models/lite-model_efficientdet_lite4_detection_metadata_1')

args = parser.parse_args()
MDL_PATH = args.model
IMG_PATH = args.image

if __name__ == '__main__':
    run()
