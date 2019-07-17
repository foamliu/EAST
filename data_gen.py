import csv
import json
import os

import cv2 as cv
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from config import im_size

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


def load_annoataion(p):
    text_polys = []
    text_tags = []
    if not os.path.exists(p):
        return np.array(text_polys, dtype=np.float32)
    with open(p, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for line in reader:
            line = [i.strip('\ufeff') for i in line]
            x1, y1, x2, y2, x3, y3, x4, y4 = list(map(int, line[:8]))
            text_polys.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])

    return np.array(text_polys, dtype=np.float32), np.array(text_tags, dtype=np.bool)


class EastDataset(Dataset):
    def __init__(self, split):
        self.split = split

        filename = '{}.json'.format(split)
        with open(filename, 'r') as file:
            self.samples = json.load(file)

        self.transformer = data_transforms[split]

    def __getitem__(self, i):
        sample = self.samples[i]
        im_fn = os.path.join('data', sample['img'])
        txt_fn = os.path.join('data', sample['gt'])
        img = cv.imread(im_fn)
        h, w = img.shape[:2]
        ratio_x = float(im_size) / w
        ratio_y = float(im_size) / h
        img = img[..., ::-1]  # RGB
        img = transforms.ToPILImage()(img)
        img = self.transformer(img)
        text_polys = load_annoataion(txt_fn)
        img = cv.resize(img, dsize=(im_size, im_size))

        # print(text_polys.type.name)

        for i in range(len(text_polys)):
            for j in range(4):
                text_polys[i][j][0] *= ratio_x
                text_polys[i][j][1] *= ratio_y

        return img, text_polys

    def __len__(self):
        return len(self.samples)


if __name__ == "__main__":
    dataset = EastDataset('train')
    print(dataset[0][1])
