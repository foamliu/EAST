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
