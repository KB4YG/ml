<div align="center">

  <img src="assets/logo.png" alt="logo" width="200" height="auto" />
  <h1>KB4YG</h1>
  
  <p>
    Project Description: Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. 
  </p>
  
  <!-- Badges -->
<p>
  <a href="https://github.com/KB4YG/ml/graphs/contributors">
    <img src="https://img.shields.io/github/contributors/KB4YG/ml" alt="contributors" />
  </a>
  <a href="">
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
    <a href="https://github.com/KB4YG/ml/iot">IoT</a>
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
This repo contains all the code for running our Machine learning model. You'll find the code for the object detection in `img_classifier.py` and a commandline interface in `detect.py`.

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
  python detect.py --image images/parking-lot-1.jpg --model-metadata models/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29
```

### From function
`image_classifier.py`

<details>
  <summary>Args</summary>
  <ul>
    <li>IMG_PATH #(REQUIRED) Path to .png or .jpg image</li>
    <li>MODEL_PATH #(REQUIRED) Path to model directory, should contain detect.tflite file</li>
    <li>MIN_CONF_LEVEL #(OPTIONAL) minimum confidence level to accept (float 0-1), default 0.5</li>
    <li>GRAPH_NAME #(OPTIONAL) name of .tflite file, default detect.tflite</li>
    <li>LABELMAP_NAME #(OPTIONAL) name of label file, default labelmap.txt</li>
    <li>SAVE_IMG #(OPTIONAL) Where or not to save image with detection boxes, default False </li>
    <li>COORDS #(OPTIONAL) Where or not to return coordinates of detect object, default False </li>
  </ul>
</details>

```python
  from image_classifier import image_classifier
  image_classifier(model_path, image_path)
```

<!-- License -->
## :warning: License

!! TODO


<!-- Contact -->
## :handshake: Contact
!! TODO

Your Name - [@twitter_handle](https://twitter.com/twitter_handle) - email@email_client.com

Project Link: [https://github.com/Louis3797/awesome-readme-template](https://github.com/Louis3797/awesome-readme-template)


<!-- Acknowledgments -->
## :gem: Acknowledgements

 - [Shields.io](https://shields.io/)
 - [Readme Template](https://github.com/othneildrew/Best-README-Template)
 - [TF-Lite Tutorial](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py)
 - [TF on RPi Tutorial](https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi)
