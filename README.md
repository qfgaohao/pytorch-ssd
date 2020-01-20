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
pip install opencv-python opencv-contrib-python numpy tqdm lxml pandas boto3
# Pytorch (with CUDA 10)
pip install torch torchvision
# Pytorch (without CUDA)
pip install torch==1.3.1+cpu torchvision==0.4.2+cpu -f https://download.pytorch.org/whl/torch_stable.html
```

Alternatively, you can install everything at once by simply running:
```
pip install -r requirements.txt
# Optionally, the development packages can be installed
pip install -r requirements-dev.txt
```

## pipenv
To instal those packages using [pipenv](https://github.com/pypa/pipenv), first install pipenv on your system:
```
pip install pipenv
```
and from the root directory, run:
```
pipenv install
```
To install with development packages:
```
pipenv install --dev
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