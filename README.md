# Single Shot MultiBox Detector Implementation in Pytorch

This repo implements [SSD (Single Shot MultiBox Detector)](https://arxiv.org/abs/1512.02325). The implementation is heavily influenced by the project [ssd.pytorch](https://github.com/amdegroot/ssd.pytorch) and [Detectron](https://github.com/facebookresearch/Detectron).
The design goal is modularity and extensibility.

Currently, it has mobilenet based SSD and VGG based SSD.

## Dependencies
1. Python 3.5+
2. OpenCV
3. Pytorch 0.4+
4. Caffe2

## Run the demo
### Run the live Mobilenet SSD demo

```bash
wget -P models https://storage.googleapis.com/models-hao/mobilenetv1-ssd-with-relu-loss-2.94.pth
python run_ssd_live_demo.py mobilenet-v1-ssd models/mobilenetv1-ssd-with-relu-loss-2.94.pth
```
### Run the live demo in Caffe2

```bash
wget -P models https://storage.googleapis.com/models-hao/mobilenet_v1_ssd_caffe2/mobilenet-v1-ssd_init_net.pb
wget -P models https://storage.googleapis.com/models-hao/mobilenet_v1_ssd_caffe2/mobilenet-v1-ssd_predict_net.pb
python run_ssd_live_caffe2.py models/mobilenetv1_ssd_init_net.pb models/mobilenetv1_ssd_predict_net.pb
```

## Pretrained Models

### Mobilenet V1 SSD

```
Average Precision Per-class:
aeroplane: 0.6742489426027927
bicycle: 0.7913672875238116
bird: 0.612096015101108
boat: 0.5616407126931772
bottle: 0.3471259064860268
bus: 0.7742298893362103
car: 0.7284171192326804
cat: 0.8360675520354323
chair: 0.5142295855384792
cow: 0.6244090341627014
diningtable: 0.7060035669312754
dog: 0.7849252606216821
horse: 0.8202146617282785
motorbike: 0.793578272243471
person: 0.7042670984734087
pottedplant: 0.40257147509774405
sheep: 0.6071252282334352
sofa: 0.7549120254763918
train: 0.8270992920206008
tvmonitor: 0.6459903029666852

Average Precision Across All Classes:0.6755
```

## Training

```bash
wget -P models https://storage.googleapis.com/models-hao/mobilenet_v1_with_relu_69_5.pth
python train_ssd.py --datasets ~/data/VOC0712/VOC2007/ ~/data/VOC0712/VOC2012/ --validation_dataset ~/data/VOC0712/test/VOC2007/ --net mobilenet-v1-ssd --base_net models/mobilenet_v1_with_relu_69_5.pth  --batch_size 24 --num_epochs 200 --scheduler cosine --lr 0.01 --t_max 200
```


The dataset path is the parent directory of the folders: Annotations, ImageSets, JPEGImages, SegmentationClass and SegmentationObject. You can use multiple datasets to train.


## Evaluation

```bash
python eval_ssd.py --net mobilenet-v1-ssd  --dataset ~/data/VOC0712/test/VOC2007/ --trained_model mobilenet-v1-ssd models/mobilenetv1-ssd-with-relu-loss-2.94.pth
```

TODO

1. Modify VGG to make it ONNX friendly.
2. Resnet 34 Based Model.
3. BatchNorm Fusion.
