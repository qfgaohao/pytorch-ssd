# Single Shot MultiBox Detector Implementation in Pytorch

This repo implements [SSD (Single Shot MultiBox Detector)](https://arxiv.org/abs/1512.02325). The implementation is heavily influenced by the project [ssd.pytorch](https://github.com/amdegroot/ssd.pytorch) and [Detectron](https://github.com/facebookresearch/Detectron).
The design goal is modularity and extensibility.

Currently, it has mobilenet based SSD and VGG based SSD.

## Dependencies
1. Python 3.5+
2. OpenCV
3. Pytorch 0.4+
4. Caffe2

## SSD 

### Run the demo
#### Run the live Mobilenet SSD demo

```bash
wget -P models https://storage.googleapis.com/models-hao/mobilenetv1-ssd-with-relu-loss-2.94.pth
python run_ssd_live_demo.py mobilenet-v1-ssd models/mobilenetv1-ssd-with-relu-loss-2.94.pth
```
#### Run the live demo in Caffe2

```bash
wget -P models https://storage.googleapis.com/models-hao/mobilenet_v1_ssd_caffe2/mobilenet-v1-ssd_init_net.pb
wget -P models https://storage.googleapis.com/models-hao/mobilenet_v1_ssd_caffe2/mobilenet-v1-ssd_predict_net.pb
python run_ssd_live_caffe2.py models/mobilenetv1_ssd_init_net.pb models/mobilenetv1_ssd_predict_net.pb
```

### Training
1. Download VOC2007 trainval and test datasets, and VOC2012 trainval datasets.
2. Download the pre-trained basenet [VGG Net](https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth) to the directory models.
3. Run **"python train_ssd.py --datasets ~/data/VOC2007/ ~/data/VOC2012/ --validation_dataset ~/data/test/VOC2007/"**. 
The dataset path is the parent directory of the folders: Annotations, ImageSets, JPEGImages, SegmentationClass and SegmentationObject. You can use multiple datasets to train.

### Evaluation

```bash
python eval_ssd.py --trained_model models/VGG-SSD-Epoch-115-Loss-2.82-map-0.768.pth  --dataset ~/data/test/VOC2007/
```
