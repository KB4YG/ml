<div align="center">

  <img src="https://raw.githubusercontent.com/KB4YG/kb4yg.github.io/main/assets/icon-white.png" alt="logo" width="200" height="auto" />
  <h1>KB4YG</h1>
  
  <p>
    Oak Creek Valley is a very popular destination for hiking and recreation for the city of Corvallis. Accessible forests in the Oak Creek Valley include the McDonald Forest, Cardwell Hill, Fitton Green, Bald Hill Farm, and others. These natural areas are enjoyed by many for hiking, mountain biking, and more. Our project, Know Before You Go, is an Internet of Things platform with a mobile app to help park visitors determine how busy a recreation site is before they arrive. By providing park visitors with this insight, we alleviate traffic congestion at trailheads, saving park visitors time and preventing overuse of natural areas.
  </p>
  
  <!-- Badges -->
<p>
  <a href="https://github.com/KB4YG/ml/graphs/contributors">
    <img src="https://img.shields.io/github/contributors/KB4YG/ml" alt="contributors" />
  </a>
  <a href="https://github.com/KB4YG/ml/commits">
    <img src="https://img.shields.io/github/last-commit/KB4YG/ml" alt="last update" />
  </a>
  <a href="https://github.com/KB4YG/ml/stargazers">
    <img src="https://img.shields.io/github/stars/KB4YG/ml" alt="stars" />
  </a>
  <a href="https://github.com/KB4YG/ml/issues/">
    <img src="https://img.shields.io/github/issues/KB4YG/ml" alt="open issues" />
  </a>
</p>
   
<h4>    
    <a href="https://kb4yg.github.io">Demo</a>
  <span> · </span>
    <a href="https://github.com/KB4YG/frontend">Frontend</a>
  <span> · </span>
    <a href="https://github.com/KB4YG/ml">ML</a>
  <span> · </span>
    <a href="https://github.com/KB4YG/iot">IoT</a>
  </h4>
</div>

<br />

<!-- Table of Contents -->
# :notebook_with_decorative_cover: Table of Contents

- [Project Overview](#star2-about-the-project)
  * [Screenshots](#camera-screenshots)
  * [Tech Stack](#space_invader-tech-stack)
- [Getting Started](#toolbox-getting-started)
  * [Prerequisites](#bangbang-prerequisites)
  * [Running Tests](#test_tube-running-tests)
- [Usage](#eyes-usage)
- [FAQ](#grey_question-faq)
- [License](#warning-license)
- [Contact](#handshake-contact)
- [Acknowledgements](#gem-acknowledgements)


<!-- About the Project -->
## :star2: About the Project
This repo contains all the code for running our Machine learning model. You'll find the code for the object detection in `img_classifier.py` and a commandline interface in `detect.py`. For a deeper dive into the code and how to train your own ML model check out the [repo wiki](https://github.com/KB4YG/ml/wiki), otherwise install/usage instructions are below.

<!-- Screenshots -->
### :camera: Screenshots

<div align="center"> 
  <img src="https://i.imgur.com/Cse10ww.png" alt="screenshot" width="600px"/>
</div>

<!-- TechStack -->
### :space_invader: Tech Stack

<li><a href="https://www.tensorflow.org/lite">Tensorflow Lite</a></li>
<li><a href="https://roboflow.com">Roboflow</a></li>
<li><a href="https://opencv.org">OpenCV</a></li>


<!-- Getting Started -->
## 	:toolbox: Getting Started

<!-- Prerequisites -->
### :bangbang: Prerequisites

There are a view dependecies that can be tricky to install. Tensorflow lite is one of them. Tested on linux, pythom 3.9

```bash
# Requires the latest pip
pip install --upgrade pip

#install cv2 for image processing
pip install opencv-python
pip install numpy

# Install tensorflow
pip install tensorflow
pip install tflite_support>=0.3.0

# install local package objdetection must clone and cd into the repo
pip install -e .
```
   
<!-- Running Tests -->
### :test_tube: Running Tests

To run tests, run the following command

```bash
  pytest 
```

<!-- Usage -->
## :eyes: Usage

### From Commandline

<details>
  <summary>Flags</summary>
  <ul>
    <li>--image # Path to .png or .jpg image</li>
    <li>--model # Path to model directory, should contain detect.tflite file</li>
  </ul>
</details>

```bash
  python detect.py --image {FULL_IMG_PATH} --model coco_ssd_mobilenet_v1_1.0_quant_2018_06_29
```

### From function
`image_classifier.py`

<details>
  <summary>Args</summary>
  <ul>
    <li>IMG_PATH #(REQUIRED) Path to .png or .jpg image</li>
    <li>MODEL_NAME #(REQUIRED) Name of one of the models listed in the `obj_detection/models` directory</li>
    <li>MIN_CONF_LEVEL #(OPTIONAL) minimum confidence level to accept (float 0-1), default 0.5</li>
    <li>GRAPH_NAME #(OPTIONAL) name of .tflite file, default detect.tflite</li>
    <li>LABELMAP_NAME #(OPTIONAL) name of label file, default labelmap.txt</li>
    <li>SAVED_IMG_PATH #(OPTIONAL) Where or not to save image with detection boxes, default null </li>
    <li>COORDS #(OPTIONAL) Where or not to return coordinates of detect object, default False </li>
  </ul>
</details>

```python
  from obj_detection import objDetection
  
  result = objDetection(model_name, img_path)
  print("Number of vehicles: ", result["vehicles"])
  print("Number of pedestrians: ", result["pedestrians"])
  print("Number of objects: ", result["objects"])
  print("Error: ", result["error"])
```


<!-- License -->
## :warning: License

Distributed under the GPL-3.0 license. See [LICENSE.txt](https://github.com/KB4YG/ml/blob/main/LICENSE) for more information.

<!-- Contact -->
## :handshake: Contact
!! TODO

Your Name - [@twitter_handle](https://twitter.com/twitter_handle) - email@email_client.com



<!-- Acknowledgments -->
## :gem: Acknowledgements

 - [Shields.io](https://shields.io/)
 - [Readme Template](https://github.com/Louis3797/awesome-readme-template)
 - [TF-Lite Tutorial](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py)
 - [TF on RPi Tutorial](https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi)
