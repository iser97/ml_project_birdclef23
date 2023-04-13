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

Additional packages shoud be installed using the following command.

```bash
pip install efficientnet_pytorch
pip install prefetch-generator
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

# Optuna Parameter Tuning

```bash
CUDA_VISIBLE_DEVICES=3 python birdclef23-optuna.py --experiment_name beats --model_name beats --eval_step 1

CUDA_VISIBLE_DEVICES=2 python birdclef23-optuna.py --experiment_name ast --model_name ast --eval_step 1
```


# TODO:
1. - [ ] Repeat the these codes (these codes are from [here](https://www.kaggle.com/code/awsaf49/birdclef23-pretraining-is-all-you-need-train/notebook#Data-Augmentation-%F0%9F%8C%88)) by Pytorch 
2. - [x] Use the [unilm pretrain model](https://github.com/microsoft/unilm/tree/master/beats)   (completed on 7/4/2023)
3. - [ ] ~~Use the [cav-mae pretrain model](https://github.com/yuangongnd/cav-mae) (Pending)~~
4. - [ ] Use Wav2vec, Audio Spectrogram Transformer, Musicnn to do the classification  (Plan to complete on 8/4/2023)
     - [x] Add Audio Spectrogram Transformer mudule (Completed on 8/4/2023)
5. - [x] Train and Valid the performation of Unilm BEATs model (run birdclef23-unilm-finetune.ipynb) (Plan to complete on 8/4/2023)
     - [x] Training Unilm BEATs model (running on 8/4/2023)
6. - [x] Add Optuna parameter adjust in the Training process. (completed on 8/4/2023)
     - [x] Merge BEATs model and AST model in the same training pipline
7. - [x] Add audio data cache in Dataset (improve the speed of loading data). (completed on 10/4/2023)
8. - [x] Add EfficientNet Model in the model hub. (completed on 10/4/2023)
9. - [x] Add BCE loss in all models (when use BCE, the model output need to use nn.Sigmoid()) (Completed on 13/4/2023)
10. - [x] Add a hyper parameter $ast_fix_layer$ to assign layers that need to be fixed (Completed on 13/4/2023)



# Reference

Some of the code in this repo is taken from [there](https://www.kaggle.com/code/awsaf49/birdclef23-pretraining-is-all-you-need-train/notebook#Data-Augmentation-%F0%9F%8C%88).