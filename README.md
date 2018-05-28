# Pytorch Computer Vision

This repo aims to implement object detection and some other common computer vision algorithms.
The design goal is modularity and extensibility.

Currently, it only implements [SSD (Single Shot MultiBox Detector)](https://arxiv.org/abs/1512.02325).


## Dependencies
1. Python 3.5+
2. Opencv
3. Pytorch 0.4+

## SSD 

The implementation is heavily influenced by the project [ssd.pytorch](https://github.com/amdegroot/ssd.pytorch) and [Detectron](https://github.com/facebookresearch/Detectron).

### Run the demo

#### Run the live demo
1. Download trained model to the directory models.
  Currently two models are provided:
  [VGG-SSD-mAP-0.768](https://storage.googleapis.com/models-hao/VGG-SSD-Epoch-115-Loss-2.82-map-0.768.pth)
and the model translated from [ssd.pytorch](https://github.com/amdegroot/ssd.pytorch) [VGG-SSD-mAP-0.774](https://s3.amazonaws.com/amdegroot-models/ssd300_mAP_77.43_v2.pth).
2. Run **"python run_ssd_live_demo.py models/VGG-SSD-Epoch-115-Loss-2.82-map-0.768.pth"**

#### Test on images

The script is only for demonstration. Feel free to change it.

##### Using Standard NMS


python run_ssd_example.py models/VGG-SSD-Epoch-115-Loss-2.82-map-0.768.pth  hard vision/test/assets/000001.jpg 

The output is ./annotated-output.jpg

##### Using Soft NMS

python run_ssd_example.py models/VGG-SSD-Epoch-115-Loss-2.82-map-0.768.pth  soft vision/test/assets/000001.jpg


### Training
1. Download VOC2007 trainval and test datasets, and VOC2012 trainval datasets.
2. Download the pre-trained basenet [VGG Net](https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth) to the directory models.
3. Run **"python train_ssd.py --datasets ~/data/VOC2007/ ~/data/VOC2012/ --validation_dataset ~/data/test/VOC2007/"**. 
The dataset path is the parent directory of the folders: Annotations, ImageSets, JPEGImages, SegmentationClass and SegmentationObject. You can use multiple datasets to train.

### Evaluation
1. python eval_ssd.py --trained_model models/VGG-SSD-Epoch-115-Loss-2.82-map-0.768.pth  --dataset ~/data/test/VOC2007/

### TODO
1. Performance analysis
2. Network Pruning
3. Quantization
4. MobileNet SSD
5. Export ONNX models