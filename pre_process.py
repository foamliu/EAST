import json
import os
import random

from config import num_train

if __name__ == "__main__":
    images = [f for f in os.listdir('data') if f.startswith('img_') and f.endswith('jpg')]
    print('num_samples: ' + str(len(images)))

    samples = []
    for img in images:
        name = img[0:img.index('.')]
        gt = 'gt_' + name + '.txt'
        samples.append({'img': img, 'gt': gt})

    train = random.sample(samples, num_train)
    print('num_train: ' + str(len(train)))

    valid = [f for f in samples if f not in train]
    print('num_valid: ' + str(len(valid)))

    with open('train.json', 'w') as file:
        json.dump(train, file, indent=4)

    with open('valid.json', 'w') as file:
        json.dump(valid, file, indent=4)
