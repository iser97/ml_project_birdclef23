# Machine Learning Project

[kaggle birdclef23](https://www.kaggle.com/competitions/birdclef-2023)

# Environment Setup
If you want to run the code on your own device instead of using the official kaggle notebook, Recommend use docker to setup the python enviroment on your own device. To install the kaggle enviroment, can reference [here](https://github.com/Kaggle/docker-python)

Under the condition you have installed docker, you can use the following command to install docker image and run image:

```bash
docker pull gcr.io/kaggle-gpu-images/python:latest

# Run the image pre-built image from gcr.io
docker run --runtime nvidia --rm -it gcr.io/kaggle-gpu-images/python /bin/bash
```

# Unilm Pretrain

1. Download birdclef-2023 dataset

First, use the following command to donwload dataset

```bash
cd download_dataset
bash download_birdclef23.sh
mkdir /kaggle/input/birdclef-2023
cp birdclef-2023.zip /kaggle/input/birdclef-2023
unzip birdclef-2023.zip
```

2. Dwonload the pretrained model

Then, download the official unilm/beats pretrained model
```bash
cd ./pretrained_models/unilm
bash download.sh
```

3. Running

Run code:

birdclef23-unilm-finetune.ipynb


# TODO:
1. - [ ] Repeat the these codes (these codes are from [here](https://www.kaggle.com/code/awsaf49/birdclef23-pretraining-is-all-you-need-train/notebook#Data-Augmentation-%F0%9F%8C%88)) by Pytorch 
2. - [x] Use the [unilm pretrain model](https://github.com/microsoft/unilm/tree/master/beats)   (completed on 7/4/2023)
3. - [ ] ~~Use the [cav-mae pretrain model](https://github.com/yuangongnd/cav-mae) (Pending)~~
4. - [ ] Use Wav2vec, Audio Spectrogram Transformer, Musicnn to do the classification  (Plan to complete on 8/4/2023)
     - [x] Add Audio Spectrogram Transformer mudule (Completed on 8/4/2023)
5. - [ ] Train and Valid the performation of Unilm BEATs model (run birdclef23-unilm-finetune.ipynb) (Plan to complete on 8/4/2023)
     - [x] Training Unilm BEATs model (running on 8/4/2023)
6. - [ ] Add Optuna parameter adjust in the Training process.



# Reference

Some of the code in this repo is taken from [there](https://www.kaggle.com/code/awsaf49/birdclef23-pretraining-is-all-you-need-train/notebook#Data-Augmentation-%F0%9F%8C%88).