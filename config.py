import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # sets device for model and PyTorch tensors

im_size = 512

train_ratio = 0.9
num_samples = 1000
num_train = int(num_samples * 0.9)
num_valid = num_samples - num_train

# Training parameters
num_workers = 1  # for data-loading; right now, only 1 works with h5py
grad_clip = 5.  # clip gradients at an absolute value of
print_freq = 100  # print training/validation stats  every __ batches
checkpoint = None  # path to checkpoint, None if none
