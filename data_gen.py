import random

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from config import im_size


class EastDataset(Dataset):
    def __init__(self, split):
        self.split = split

        filename = '{}_names.txt'.format(split)
        with open(filename, 'r') as file:
            self.names = file.read().splitlines()

        self.transformer = data_transforms[split]

    def __getitem__(self, i):
        name = self.names[i]
        fcount = int(name.split('.')[0].split('_')[0])
        bcount = int(name.split('.')[0].split('_')[1])
        im_name = fg_files[fcount]
        bg_name = bg_files[bcount]
        img, alpha, fg, bg = process(im_name, bg_name)

        # crop size 320:640:480 = 1:1:1
        different_sizes = [(320, 320), (480, 480), (640, 640)]
        crop_size = random.choice(different_sizes)

        trimap = gen_trimap(alpha)
        x, y = random_choice(trimap, crop_size)
        img = safe_crop(img, x, y, crop_size)
        alpha = safe_crop(alpha, x, y, crop_size)

        trimap = gen_trimap(alpha)

        # Flip array left to right randomly (prob=1:1)
        if np.random.random_sample() > 0.5:
            img = np.fliplr(img)
            trimap = np.fliplr(trimap)
            alpha = np.fliplr(alpha)

        x = torch.zeros((4, im_size, im_size), dtype=torch.float)
        img = transforms.ToPILImage()(img)
        img = self.transformer(img)
        x[0:3, :, :] = img
        x[3, :, :] = torch.from_numpy(trimap.copy()) / 255.

        y = np.empty((2, im_size, im_size), dtype=np.float32)
        y[0, :, :] = alpha / 255.
        mask = np.equal(trimap, 128).astype(np.float32)
        y[1, :, :] = mask

        return x, y

    def __len__(self):
        return len(self.names)


if __name__ == "__main__":
    dataset = EastDataset('train')
