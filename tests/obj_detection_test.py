from obj_detection import objDetection, load_labels
from from_root import from_root

# valid inputs
IMG_PATH = 'tests/images/parking-lot.jpg'
MDL_PATH = 'tests/models/model-metadata'
CONF_LEVEL = 0.5


def test_detection():
    expected = {
        "error": "",
        "vehicles": 1,
        "pedestrians": 0,
        "confidence-threshold": CONF_LEVEL,
        "objects": [{'name': 'car', 'confidence': 0.73046875, 'coord': {}}],
    }

    result = objDetection(MDL_PATH, IMG_PATH)
    assert result == expected


def test_bad_img_path():
    expected = {
        "error": "Image not found, check path",
        "vehicles": -1,
        "pedestrians": -1,
        "confidence-threshold": CONF_LEVEL,
        "objects": [],
    }

    result = objDetection(MDL_PATH, ".bad_path")
    assert result == expected


def test_bad_model_path():
    expected = {
        "error": "Invalid model-metadata path",
        "vehicles": -1,
        "pedestrians": -1,
        "confidence-threshold": CONF_LEVEL,
        "objects": [],
    }

    result = objDetection(".bad_path", IMG_PATH)
    assert result == expected


def test_load_metadata_labels():
    result = load_labels(str(from_root(MDL_PATH + "/detect.tflite")), str(from_root(MDL_PATH + "/labelmap.txt")))
    assert result != []


def test_load_labelmap():
    path = "tests/models/model-no-metadata"
    result = load_labels(str(from_root(path+"/detect.tflite")), str(from_root(path+"/labelmap.txt")))
    assert result != []


def test_detection_coords():
    expected = {
        "error": "",
        "vehicles": 1,
        "pedestrians": 0,
        "confidence-threshold": CONF_LEVEL,
        "objects": [
            {'name': 'car',
             'confidence': 0.73046875,
             'coord':
                 {'bottom-left': (742, 469),
                  'bottom-right': (1241, 469),
                  'top-left': (742, 674),
                  'top-right': (1241, 674)}
             }]
    }
    result = objDetection(MDL_PATH, IMG_PATH, COORDS=True)
    assert result == expected
