# Just a wrapper to call img_classifier from the command line
# Run: python main.py --image "image.jpg"

from obj_detection import objDetection
import argparse
from from_root import from_root


def run(model, imagePath):
    img_path = imagePath
    if img_path == 'images/parking-lot-5.jpg':
        img_path = str(from_root(img_path))

    result = objDetection(model, img_path)
    print("Number of vehicles: ", result["vehicles"])
    print("Number of pedestrians: ", result["pedestrians"])
    print("Number of objects: ", result["objects"])
    print("Error: ", result["error"])


# get commandline arguments
# parser = argparse.ArgumentParser()
# parser.add_argument('--image',
#                     help='path to image file',
#                     type=str,
#                     default='images/parking-lot-5.jpg')
# parser.add_argument('--model',
#                     help='Select model to use: coco_ssd_mobilenet_v1_1.0_quant_2018_06_29, '
#                          'lite-model_efficientdet_lite2_detection_metadata_1, TFLite_model_bbd',
#                     type=str,
#                     default='coco_ssd_mobilenet_v1_1.0_quant_2018_06_29')

# args = parser.parse_args()
# MDL_PATH = args.model
# IMG_PATH = args.image

# if __name__ == '__main__':
#     run()
