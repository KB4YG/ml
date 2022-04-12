from obj_detection import objDetection, load_labels

# valid inputs
IMG_PATH = 'unit_test/parking-lot.jpg'
MDL_PATH = 'unit_test/model-metadata'
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
    result = load_labels(MDL_PATH + "/detect.tflite", MDL_PATH + "/labelmap.txt")
    assert result != []


def test_load_labelmap():
    result = load_labels("unit_test/model-no-metadata/detect.tflite", "unit_test/model-no-metadata/labelmap.txt")
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
