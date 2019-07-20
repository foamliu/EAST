# EAST: An Efficient and Accurate Scene Text Detector

This is a PyTorch re-implementation of EAST: An Efficient and Accurate Scene Text Detector.

## Dependency

- PyTorch 1.1.0

## Usage
### Data Pre-processing
Extract training images:
```bash
$ python extract.py
$ python pre_process.py
```

### Train
```bash
$ python train.py
```

If you want to visualize during training, run in your terminal:
```bash
$ tensorboard --logdir runs
```

### Demo
```bash
$ python demo.py
```

1|2|
|----|---|
|![image](https://github.com/foamliu/EAST/raw/master/images/out_0.jpg)|![image](https://github.com/foamliu/EAST/raw/master/images/out_1.jpg)|
|![image](https://github.com/foamliu/EAST/raw/master/images/out_2.jpg)|![image](https://github.com/foamliu/EAST/raw/master/images/out_3.jpg)|
|![image](https://github.com/foamliu/EAST/raw/master/images/out_4.jpg)|![image](https://github.com/foamliu/EAST/raw/master/images/out_5.jpg)|
|![image](https://github.com/foamliu/EAST/raw/master/images/out_6.jpg)|![image](https://github.com/foamliu/EAST/raw/master/images/out_7.jpg)|
|![image](https://github.com/foamliu/EAST/raw/master/images/out_8.jpg)|![image](https://github.com/foamliu/EAST/raw/master/images/out_9.jpg)|
