import datetime

import cv2
import requests
import os
from img_classifier import imgClassify
import argparse

def capturePhoto():
    return "test_img/parking-lot-3.jpeg" #TODO TESTING
    videoCaptureObject = cv2.VideoCapture(0)
    result = True
    while (result):
        ret, frame = videoCaptureObject.read()
        ts = datetime.datetime.now()
        img_path = ts.strftime("%Y/%m/%d, %H:%M:%S")
        cv2.imwrite(img_path, frame)
        result = False
    videoCaptureObject.release()
    cv2.destroyAllWindows()
    return img_path


def deleteImage(img_path):
    try:
        os.remove(img_path)
    except:
        pass


def putDatabase(count):
    vehicle_count = -1
    if 'car' in count:
        vehicle_count = count['car']

    r = requests.put(API_URL, data={'ParkingLocation': LOCATION, 'OpenSpaces': vehicle_count})
    if r.status_code == 200:
        return True
    print("HTTP ERROR Status code:", r.status_code)
    return False


def run():
    img_path = capturePhoto()
    _, count = imgClassify(MDL_PATH, img_path)
    putDatabase(count)
    # deleteImage(img_path) #TODO TESTING

# get commandline arguments
parser = argparse.ArgumentParser()
parser.add_argument('--location',
                    help='Name of park, must match database',
                    type=str,
                    default="Fitton Green",
                    required=True)
parser.add_argument('--model',
                    help='Path to model folder',
                    type=str,
                    default='models/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29')
parser.add_argument('--url',
                    help='Minimum confidence threshold for displaying detected objects',
                    default="https://cfb32cwake.execute-api.us-west-2.amazonaws.com/default/",
                    type=str,
                    required=True)

args = parser.parse_args()

MDL_PATH = args.model
LOCATION = args.location
API_URL = args.url

# Other arguments?
# api authentication key
# Debug mode or flag to save images for future algorithm

if __name__ == '__main__':
    run()
