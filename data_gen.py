import os

import cv2 as cv
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

from config import im_size, training_data_path
from icdar import load_annoataion, get_images

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.ColorJitter(0.5, 0.5, 0.5, 0.25),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'valid': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


class EastDataset(Dataset):
    def __init__(self, split):
        self.split = split

        self.image_list = np.array(get_images())
        print('{} training images in {}'.format(
            self.image_list.shape[0], training_data_path))

        self.transformer = data_transforms[split]

    def __getitem__(self, i):
        sample = self.samples[i]
        im_fn = os.path.join('data', sample['img'])
        txt_fn = os.path.join('data', sample['gt'])
        img = cv.imread(im_fn)
        h, w = img.shape[:2]
        ratio_x = float(im_size) / w
        ratio_y = float(im_size) / h
        img = cv.resize(img, dsize=(im_size, im_size))
        img = img[..., ::-1]  # RGB
        img = transforms.ToPILImage()(img)
        img = self.transformer(img)
        text_polys = load_annoataion(txt_fn)
        text_polys[:, :, 0] *= ratio_x
        text_polys[:, :, 1] *= ratio_y

        print(img.size())
        print(text_polys.shape)

        return img, text_polys

    def __len__(self):
        return len(self.image_list)


if __name__ == "__main__":
    dataset = EastDataset('train')
    print(dataset[0][1])
