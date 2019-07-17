import json
import os

import cv2 as cv
from torch.utils.data import Dataset
from torchvision import transforms

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

        filename = '{}.json'.format(split)
        with open(filename, 'r') as file:
            self.samples = json.load(file)

        self.transformer = data_transforms[split]

    def __getitem__(self, i):
        sample = self.samples[i]
        filename = os.path.join('data', sample['img'])
        img = cv.imread(filename)
        img = img[..., ::-1]  # RGB
        img = transforms.ToPILImage()(img)
        img = self.transformer(img)
        label = sample['gt']

        return img, label

    def __len__(self):
        return len(self.samples)


if __name__ == "__main__":
    dataset = EastDataset('train')
    print(dataset[0][1])
