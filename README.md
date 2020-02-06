# new_tracking_system

This repository was created to host the new wagon tracking system code. The tracking strategy consists in detecting the front face of the wagons (the drain region) present in each frame of a video stream.

The object detector used in this project is based on the SSD (Single Shot Detector) algorithm, a deep learning algorithm that predicts both the locations of all objects pertaining to an image and their class, at the same time, in a single pass. The SSD code author is [gfgaohao](https://github.com/qfgaohao), and the original repository can be found [here](https://github.com/qfgaohao/pytorch-ssd). There, one can find the original README, where the instructions of how to train the SSD, net with other backbones than those cited here, are.

This repository also has the [labeling tool](https://github.com/Cartucho/OpenLabeling) created by [Cartucho](https://github.com/Cartucho), in case of a new dataset needs to be created. Instructions of how to use it can be found in the module's README.

# Dependencies
- Python3
- Numpy
- OpenCV (with contrib modules)
- tqdm
- lxml
- Pytorch 1.0+
- Caffe2
- Pandas
- Boto3 (to train SSD models on the Google OpenImages Dataset).

Additionally, the SSD package must be installed. [This](https://github.com/TheCamilovisk/PytorchSSD) repository is a modification of the original one by gfgaohao.

## pip
To instal those packages using pip, from the root directory run:
```
pip install opencv-python opencv-contrib-python numpy tqdm lxml pandas boto3 sortedcontainers
# Pytorch (with CUDA 10)
pip install torch torchvision
# Pytorch (without CUDA)
pip install torch==1.3.1+cpu torchvision==0.4.2+cpu -f https://download.pytorch.org/whl/torch_stable.html
```

## Install submodules
After installing the dependencies, from the root directory run:
```
git submodule init
git submodule update
```

# Quickstart
To run the detector demo, first download one of the pre-trained models.

## MobileNet V1
```
wget -P models https://storage.googleapis.com/models-thecamilowisk/mobilenet_v1.pth
```

## VGG16
```
wget -P models https://storage.googleapis.com/models-thecamilowisk/vgg16.pth
```

## Run the demo script
```
python run_detector.py <net_type> <model_filepath> resources/labels.txt [video_filepath]
```

Run `python run_detector.py --h` for more information.

Re-train the net.

```
python train_detector.py --datasets <train dataset> --validation_dataset <validation_dataset> --net <net_type> --pretrained_ssd <pretrained_model_file>  --batch_size 24 --num_epochs 200 --scheduler cosine --lr 0.01 --base_net_lr 0.001 --t_max 200
```