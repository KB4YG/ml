## Setup 

### Docker
https://github.com/ultralytics/yolov5/wiki/Docker-Quickstart
```bash
docker run --ipc=host -it -v {FILE_PATH}:/usr/src/datasets --gpus all ultralytics/yolov5:latest
```
^ mounts dataset path and enables GPU acess for faster training

### Codlab 
https://colab.research.google.com/github/roboflow-ai/yolov5-custom-training-tutorial/blob/main/yolov5-custom-training.ipynb#scrollTo=eaFNnxLJbq4J




## Train custom ML model using pytorch and YOLOv5

1) Create Dataset roboflow...
take or find vehicle images for create a special dataset for fine-tuning.

File Distribution
Train : 70%
Validition : 20%
Test : 10%

Below is unnessasry if you export as YOLO from roboflow, will auto gen
## dataset.yaml

config dataset.yaml for the address and information of your dataset.

```
path: Dataset/dataset-vehicles  # dataset root dir
train: images/train  # train images (relative to 'path')
val: images/val  # val images (relative to 'path')
test:  # test images (optional)

# Classes
nc: 5  # number of classes
names: [ 'Car', 'Motorcycle', 'Truck', 'Bus', 'Bicycle']  # class names

```
## Train

2) fine-tuning on a pre-trained model of yolov5.

img: define input image size
batch: determine batch size
epochs: define the number of training epochs. (Note: often, 3000+ are common here!)
data: Our dataset locaiton is saved in the dataset.location
weights: specify a path to weights to start transfer learning from. Here we choose the generic COCO pretrained checkpoint.
cache: cache images for faster training - uses a lot of ram

Diffrent types of yolov5?.pt are available s Small, m Medium, and x Large. As size increased so does accuracy and look up time.
```
python train.py --img 640 --batch 16 --epochs 50 --data dataset.yaml --weights yolov5m.pt
#or
python train.py --img 640 --batch 16 --epochs 150 --data {dataset.location}/data.yaml --weights yolov5s.pt
```

## Test

after train, gives you weights of train and you should use them for test.

```
python detect.py --weights runs/train/exp12/weights/best.pt --source test_images/imtest13.JPG
```

## Export
*** Switch yolov5s.pt with the output of your train.py i.e /runs/train/...
python detect.py --weights yolov5s.pt --source path/to/images  # run inference on images and videos
python export.py --weights yolov5s.pt --include coreml tflite  # export models to other formats