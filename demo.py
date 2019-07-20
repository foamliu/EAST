import os
import random

import cv2 as cv
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

from config import device
from data_gen import data_transforms
from eval import get_images_for_test, detect, resize_image, sort_poly
from icdar import polygon_area
from utils import ensure_folder

if __name__ == "__main__":
    checkpoint = 'BEST_checkpoint.tar'
    checkpoint = torch.load(checkpoint)
    model = checkpoint['model']
    model = model.to(device)
    model.eval()

    output_dir = 'images'
    ensure_folder(output_dir)

    transformer = data_transforms['test']

    im_fn_list = get_images_for_test()
    im_fn_list = random.sample(im_fn_list, 10)

    for idx in tqdm(range(len(im_fn_list))):
        im_fn = im_fn_list[idx]
        im = cv.imread(im_fn)
        im = im[..., ::-1]  # RGB

        im_resized, (ratio_h, ratio_w) = resize_image(im)
        im_resized = transforms.ToPILImage()(im_resized)
        im_resized = transformer(im_resized)
        im_resized = im_resized.to(device)
        im_resized = im_resized.unsqueeze(0)

        timer = {'net': 0, 'restore': 0, 'nms': 0}

        score, geometry = model(im_resized)

        score = score.permute(0, 2, 3, 1)
        geometry = geometry.permute(0, 2, 3, 1)
        score = score.data.cpu().numpy()
        geometry = geometry.data.cpu().numpy()

        boxes, timer = detect(score_map=score, geo_map=geometry, timer=timer)
        # print('EAST <==> TEST <==> idx:{} <==> restore:{:.2f}ms'.format(idx, timer['restore'] * 1000))
        # print('EAST <==> TEST <==> idx:{} <==> nms    :{:.2f}ms'.format(idx, timer['nms'] * 1000))

        # print('EAST <==> TEST <==> Record and Save <==> id:{} <==> Begin'.format(idx))
        if boxes is not None:
            boxes = boxes[:, :8].reshape((-1, 4, 2))
            boxes[:, :, 0] /= ratio_w
            boxes[:, :, 1] /= ratio_h

        if boxes is not None:
            res_file = os.path.join(output_dir, 'res_{}.txt'.format(idx))
            with open(res_file, 'w') as f:
                for box in boxes:
                    box = sort_poly(box.astype(np.int32))

                    if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                        # print('wrong direction')
                        continue

                    if box[0, 0] < 0 or box[0, 1] < 0 or box[1, 0] < 0 or box[1, 1] < 0 or box[2, 0] < 0 or box[
                        2, 1] < 0 or box[3, 0] < 0 or box[3, 1] < 0:
                        continue

                    poly = np.array([[box[0, 0], box[0, 1]], [box[1, 0], box[1, 1]], [box[2, 0], box[2, 1]],
                                     [box[3, 0], box[3, 1]]])

                    p_area = polygon_area(poly)
                    if p_area > 0:
                        poly = poly[(0, 3, 2, 1), :]

                    f.write(
                        '{},{},{},{},{},{},{},{}\r\n'.format(poly[0, 0], poly[0, 1], poly[1, 0], poly[1, 1], poly[2, 0],
                                                             poly[2, 1], poly[3, 0], poly[3, 1], ))
                    cv.polylines(im[:, :, ::-1], [box.astype(np.int32).reshape((-1, 1, 2))], True, color=(255, 255, 0),
                                 thickness=1)

        save_img_path = os.path.join(output_dir, 'out_{}.jpg'.format(idx))
        cv.imwrite(save_img_path, im[:, :, ::-1])
