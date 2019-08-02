# EAST: An Efficient and Accurate Scene Text Detector

## Introduction

This is a PyTorch re-implementation of EAST: An Efficient and Accurate Scene Text Detector ([paper](https://arxiv.org/abs/1704.03155)). 

The features are summarized blow:
- Only RBOX part is implemented.
- A fast Locality-Aware NMS in C++ provided by the paper's author.
- The pre-trained model provided achieves 81.61 F1-score on ICDAR 2015 Incidental Scene Text Detection Challenge using only training images from ICDAR 2015 and 2013. see here for the detailed results.
- Differences from original paper
    - Use ResNet-50 rather than PVANET
    - Use dice loss (optimize IoU of segmentation) rather than balanced cross entropy
    - Use linear learning rate decay rather than staged learning rate decay

## Performance

### ICDAR 2015 

|Model|Recall|Precision|Hmean|Download|
|---|---|---|---|---|
|PyTorch re-implementation of EAST|74.48%|90.26%|81.61%|[Link](https://github.com/foamliu/EAST/releases/download/v1.0/BEST_checkpoint.tar)

![image](https://github.com/foamliu/EAST/raw/master/images/Results_IoU.png)

[Link](https://rrc.cvc.uab.es/?ch=4&com=evaluation&task=1)

### Offline evaluation

```bash
$ python eval.py
$ ./eval.sh

```

## Credit
Most codes are ported from [argman/EAST](https://github.com/argman/EAST) (the Tensorflow re-implementation).

## DataSet

Model is trained & tested on [ICDAR 2015](http://rrc.cvc.uab.es/?ch=4&com=downloads). Please download following 4 files then put them under "data" folder:
- ch4_training_images.zip
- ch4_training_localization_transcription_gt.zip
- ch4_test_images.zip
- Challenge4_Test_Task1_GT.zip


## Dependency

- PyTorch 1.1.0

## Usage
### Data Pre-processing
Extract training & test images:
```bash
$ python extract.py
```

### Train
```bash
$ python train.py
```

If you want to visualize during training, run in your terminal:
```bash
$ tensorboard --logdir runs
```

![image](https://github.com/foamliu/EAST/raw/master/images/train_loss.png)
![image](https://github.com/foamliu/EAST/raw/master/images/hmean.png)

### Demo
Pick 10 random test examples from ICDAR-2015:
```bash
$ python demo.py
```

Examples|
|----|
|![image](https://github.com/foamliu/EAST/raw/master/images/out_0.jpg)
|![image](https://github.com/foamliu/EAST/raw/master/images/out_1.jpg)
|![image](https://github.com/foamliu/EAST/raw/master/images/out_2.jpg)
|![image](https://github.com/foamliu/EAST/raw/master/images/out_3.jpg)
|![image](https://github.com/foamliu/EAST/raw/master/images/out_4.jpg)
|![image](https://github.com/foamliu/EAST/raw/master/images/out_5.jpg)
|![image](https://github.com/foamliu/EAST/raw/master/images/out_6.jpg)
|![image](https://github.com/foamliu/EAST/raw/master/images/out_7.jpg)
|![image](https://github.com/foamliu/EAST/raw/master/images/out_8.jpg)
|![image](https://github.com/foamliu/EAST/raw/master/images/out_9.jpg)
