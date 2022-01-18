
### Benchmark results
```
parking-lot-1.jpg > 22 cars
	lite-model_efficientdet_lite2_detection_metadata_1
		 {'car': 5}
		 [('car', 85), ('car', 74), ('car', 72), ('car', 67), ('car', 67)]

	coco_ssd_mobilenet_v1_1.0_quant_2018_06_29
		 {'car': 10}
		 [('car', 82), ('car', 62), ('car', 60), ('car', 59), ('car', 59), ('car', 58), ('car', 58), ('car', 57), ('car', 56), ('car', 56)]
```
```
parking-lot-2.jpg 5 cars
	lite-model_efficientdet_lite2_detection_metadata_1
		 {'car': 2}
		 [('car', 67), ('car', 56)]

	coco_ssd_mobilenet_v1_1.0_quant_2018_06_29
		 {'car': 4}
		 [('car', 64), ('car', 60), ('car', 56), ('car', 55)]
```
```
parking-lot-3.jpeg 1 car
	lite-model_efficientdet_lite2_detection_metadata_1
		 {'truck': 1}
		 [('truck', 69)]

	coco_ssd_mobilenet_v1_1.0_quant_2018_06_29
		 {'car': 1}
		 [('car', 73)]
```
```
parking-lot-4.jpg 7 cars
	lite-model_efficientdet_lite2_detection_metadata_1
		 {'car': 5}
		 [('car', 80), ('car', 75), ('car', 72), ('car', 64), ('car', 60)]

	coco_ssd_mobilenet_v1_1.0_quant_2018_06_29
		 {'car': 5}
		 [('car', 71), ('car', 69), ('car', 63), ('car', 59), ('car', 57)]
```
```
parking-lot-5.png > 10 cars
	lite-model_efficientdet_lite2_detection_metadata_1
		 {'car': 6, 'person': 1}
		 [('car', 81), ('car', 71), ('car', 69), ('car', 67), ('car', 62), ('car', 62), ('person', 58)]

	coco_ssd_mobilenet_v1_1.0_quant_2018_06_29
		 {'person': 2, 'car': 8}
		 [('person', 68), ('car', 65), ('car', 64), ('car', 62), ('car', 62), ('car', 60), ('car', 58), ('car', 58), ('car', 57), ('person', 55)]
```
```
parking-lot-6.jpg > 7 cars
	lite-model_efficientdet_lite2_detection_metadata_1
		 {'truck': 1}
		 [('truck', 53)]

	coco_ssd_mobilenet_v1_1.0_quant_2018_06_29
		 {'person': 2, 'car': 6, 'truck': 2}
		 [('person', 71), ('car', 60), ('truck', 59), ('truck', 57), ('person', 56), ('car', 56), ('car', 56), ('car', 56), ('car', 55), ('car', 52)]
```
```
parking-lot-7.jpg 1 car + trailer
	lite-model_efficientdet_lite2_detection_metadata_1
		 {'car': 1}
		 [('car', 72)]

	coco_ssd_mobilenet_v1_1.0_quant_2018_06_29
		 {'car': 2}
		 [('car', 60), ('car', 53)]
```
```
parking-lot-8.jpg 6 cars
	lite-model_efficientdet_lite2_detection_metadata_1
		 {'truck': 2}
		 [('truck', 64), ('truck', 58)]

	coco_ssd_mobilenet_v1_1.0_quant_2018_06_29
		 {'car': 3}
		 [('car', 65), ('car', 57), ('car', 53)]
```

