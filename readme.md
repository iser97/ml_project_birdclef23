# Machine Learning Project

[kaggle birdclef23](https://www.kaggle.com/competitions/birdclef-2023)

```bash
birdclef23-effnet-fsr-cutmixup-train.ipynb
birdclef23-pretraining-is-all-you-need-train.ipynb
```



## Unilm Pretrain

1. 下载birdclef-2023数据集

```bash
cd download_dataset
bash download_birdclef23.sh
mkdir /kaggle/input/birdclef-2023
cp birdclef-2023.zip /kaggle/input/birdclef-2023
unzip birdclef-2023.zip
```

2. 下载预训练模型

```bash
cd ./pretrained_models/unilm
bash download.sh
```

3. 运行

birdclef23-unilm-finetune.ipynb


TODO:
1. Repeat the above code (these codes is from [here](https://www.kaggle.com/code/awsaf49/birdclef23-pretraining-is-all-you-need-train/notebook#Data-Augmentation-%F0%9F%8C%88)) by Pytorch
2. Use the [unilm pretrain model](https://github.com/microsoft/unilm/tree/master/beats)
3. Use the [cav-mae pretrain model](https://github.com/yuangongnd/cav-mae)

