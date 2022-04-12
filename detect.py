# Just a wrapper to call img_classifier from the command line
# Run: python main.py --image "image.jpg"

from obj_detection import objDetection
import argparse


def run():
    result = objDetection(MDL_PATH, IMG_PATH, SAVE_IMG=True)
    print("Number of vehicles: ", result["vehicles"])
    print("Number of pedestrians: ", result["pedestrians"])
    print("Number of objects: ", result["objects"])


# get commandline arguments
parser = argparse.ArgumentParser()
parser.add_argument('--image',
                    help='image to classify',
                    type=str,
                    default='test_img/parking-lot-5.jpg')
parser.add_argument('--model-metadata',
                    help='Path to model-metadata folder',
                    type=str,
                    default='models/detect_21k')

args = parser.parse_args()
MDL_PATH = args.model
IMG_PATH = args.image

if __name__ == '__main__':
    run()
