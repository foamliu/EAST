import os

import cv2 as cv
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from config import input_size, training_data_path, test_data_path, background_ratio, random_scale, geometry
from icdar import load_annoataion, get_images, check_and_validate_polys, crop_area, generate_rbox

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.ColorJitter(0.5, 0.5, 0.5, 0.25),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


def collate_fn(batch):
    img, score_map, geo_map, training_mask = zip(*batch)  # tuple
    bs = len(score_map)
    images = []
    score_maps = []
    geo_maps = []
    training_masks = []
    for i in range(bs):
        if img[i] is not None:
            # a = torch.from_numpy(img[i])
            a = img[i]
            images.append(a)

            b = torch.from_numpy(score_map[i])
            b = b.permute(2, 0, 1)
            score_maps.append(b)

            c = torch.from_numpy(geo_map[i])
            c = c.permute(2, 0, 1)
            geo_maps.append(c)

            d = torch.from_numpy(training_mask[i])
            d = d.permute(2, 0, 1)
            training_masks.append(d)

    images = torch.stack(images, 0)
    score_maps = torch.stack(score_maps, 0)
    geo_maps = torch.stack(geo_maps, 0)
    training_masks = torch.stack(training_masks, 0)

    return images, score_maps, geo_maps, training_masks


def get_data_record(image_list, i, data_path, transformer):
    im_fn = image_list[i]
    im = cv.imread(im_fn)
    # print im_fn
    h, w, _ = im.shape
    txt_fn = im_fn.replace(data_path, '')
    txt_fn = os.path.join(data_path, 'gt_' + txt_fn.split('.')[0] + '.txt')
    assert (os.path.exists(txt_fn))

    text_polys, text_tags = load_annoataion(txt_fn)

    text_polys, text_tags = check_and_validate_polys(text_polys, text_tags, (h, w))
    # if text_polys.shape[0] == 0:
    #     continue
    # random scale this image
    rd_scale = np.random.choice(random_scale)
    im = cv.resize(im, dsize=None, fx=rd_scale, fy=rd_scale)
    text_polys *= rd_scale
    # print rd_scale
    # random crop a area from image
    if np.random.rand() < background_ratio:
        # crop background
        im, text_polys, text_tags = crop_area(im, text_polys, text_tags, crop_background=True)
        # assert (text_polys.shape[0] > 0)
        # pad and resize image
        new_h, new_w, _ = im.shape
        max_h_w_i = np.max([new_h, new_w, input_size])
        im_padded = np.zeros((max_h_w_i, max_h_w_i, 3), dtype=np.uint8)
        im_padded[:new_h, :new_w, :] = im.copy()
        im = cv.resize(im_padded, dsize=(input_size, input_size))
        score_map = np.zeros((input_size, input_size), dtype=np.uint8)
        geo_map_channels = 5 if geometry == 'RBOX' else 8
        geo_map = np.zeros((input_size, input_size, geo_map_channels), dtype=np.float32)
        training_mask = np.ones((input_size, input_size), dtype=np.uint8)
    else:
        im, text_polys, text_tags = crop_area(im, text_polys, text_tags, crop_background=False)
        assert (text_polys.shape[0] > 0)

        h, w, _ = im.shape

        # pad the image to the training input size or the longer side of image
        new_h, new_w, _ = im.shape
        max_h_w_i = np.max([new_h, new_w, input_size])
        im_padded = np.zeros((max_h_w_i, max_h_w_i, 3), dtype=np.uint8)
        im_padded[:new_h, :new_w, :] = im.copy()
        im = im_padded
        # resize the image to input size
        new_h, new_w, _ = im.shape
        resize_h = input_size
        resize_w = input_size
        im = cv.resize(im, dsize=(resize_w, resize_h))
        resize_ratio_3_x = resize_w / float(new_w)
        resize_ratio_3_y = resize_h / float(new_h)
        text_polys[:, :, 0] *= resize_ratio_3_x
        text_polys[:, :, 1] *= resize_ratio_3_y
        new_h, new_w, _ = im.shape
        score_map, geo_map, training_mask = generate_rbox((new_h, new_w), text_polys, text_tags)

    im = im[..., ::-1]  # RGB
    im = transforms.ToPILImage()(im)
    im = transformer(im)

    score_map = score_map[::4, ::4, np.newaxis].astype(np.float32)
    # score_map = np.transpose(score_map, (2, 0, 1))
    geo_map = geo_map[::4, ::4, :].astype(np.float32)
    # geo_map = np.transpose(geo_map, (2, 0, 1))
    training_mask = training_mask[::4, ::4, np.newaxis].astype(np.float32)
    # training_mask = np.transpose(training_mask, (2, 0, 1))

    return im, score_map, geo_map, training_mask  # , text_polys


class EastDataset(Dataset):
    def __init__(self, split):
        if split == 'train':
            self.data_path = training_data_path
        else:
            self.data_path = test_data_path

        self.image_list = np.array(get_images(self.data_path))
        self.transformer = data_transforms[split]

        print('{} {} images in {}'.format(
            self.image_list.shape[0], split, self.data_path))

    def __getitem__(self, i):
        idx = i
        while True:
            try:
                return get_data_record(self.image_list, idx, self.data_path, self.transformer)
            except TypeError:
                import random
                idx = random.randint(0, len(self.image_list) - 1)

    def __len__(self):
        return len(self.image_list)


if __name__ == "__main__":
    import random
    import matplotlib.pyplot as plt
    import matplotlib.patches as Patches

    dataset = EastDataset('test')
    length = len(dataset)
    index = random.randint(0, length - 1)
    print('index: ' + str(index))

    im = dataset[index][0]
    score_map = dataset[index][1][::, ::, 0]
    geo_map = dataset[index][2]
    training_mask = dataset[index][3]
    text_polys = dataset[index][4]

    fig, axs = plt.subplots(3, 2, figsize=(20, 30))

    axs[0, 0].imshow(im[..., ::-1])
    axs[0, 0].set_xticks([])
    axs[0, 0].set_yticks([])
    for poly in text_polys:
        poly_h = min(abs(poly[3, 1] - poly[0, 1]), abs(poly[2, 1] - poly[1, 1]))
        poly_w = min(abs(poly[1, 0] - poly[0, 0]), abs(poly[2, 0] - poly[3, 0]))
        axs[0, 0].add_artist(Patches.Polygon(
            poly, facecolor='none', edgecolor='green', linewidth=2, linestyle='-', fill=True))
        axs[0, 0].text(poly[0, 0], poly[0, 1], '{:.0f}-{:.0f}'.format(poly_h, poly_w), color='purple')

    print(score_map.shape)
    print(np.max(score_map))
    print(np.min(score_map))

    axs[0, 1].imshow(score_map)
    axs[0, 1].set_xticks([])
    axs[0, 1].set_yticks([])
    axs[1, 0].imshow(geo_map[::, ::, 0])
    axs[1, 0].set_xticks([])
    axs[1, 0].set_yticks([])
    axs[1, 1].imshow(geo_map[::, ::, 1])
    axs[1, 1].set_xticks([])
    axs[1, 1].set_yticks([])
    axs[2, 0].imshow(geo_map[::, ::, 2])
    axs[2, 0].set_xticks([])
    axs[2, 0].set_yticks([])
    axs[2, 1].imshow(training_mask[::, ::, 0])
    axs[2, 1].set_xticks([])
    axs[2, 1].set_yticks([])
    plt.tight_layout()
    plt.show()
    plt.close()
