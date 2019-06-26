# AnimeHeadDetector
An object detector for character heads in animes, based on Yolo V3.

## Motivation

Yet another anime face/head detector, in addition to [Faster RCNN](https://github.com/qhgz2013/anime-face-detector) and [Viola Detector](https://github.com/nagadomi/lbpcascade_animeface).
The major difference is this one is more on character head instead of face only, so we also have outputs when the face is not visible.
The algorithm is based on [Yolo](https://pjreddie.com/darknet/yolo/) V3 and [Pytorch implementation](https://github.com/ayooshkathuria/pytorch-yolo-v3) so it's good for high-speed processing and the usage is straight-forward.

## Usage

* First clone the repo using `git clone --recursive https://github.com/grapeot/AnimeHeadDetector`. Note the `--recursive` switch. It's necessary because we used `git submodule` to manage the dependency of PyTorch implementation.
* Use `pip install -r requirements.txt` to install the dependencies.
* And then download the pre-trained model using the `./downloadWeights.sh` script.
* `main.py` provides all necessary functionality. `python main.py` should run out of box. Check `python main.py --help` for detailed usage.
