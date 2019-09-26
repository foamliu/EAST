import time

import torch

from models import EastModel

if __name__ == '__main__':
    checkpoint = 'BEST_checkpoint.tar'
    print('loading {}...'.format(checkpoint))
    start = time.time()
    checkpoint = torch.load(checkpoint)
    print('elapsed {} sec'.format(time.time() - start))
    model = checkpoint['model'].module
    print(type(model))

    # model.eval()
    filename = 'east.pt'
    print('saving {}...'.format(filename))
    start = time.time()
    torch.save(model.state_dict(), filename)
    print('elapsed {} sec'.format(time.time() - start))


    class HParams:
        def __init__(self):
            self.pretrained = True
            self.network = 'r50'


    config = HParams()

    print('loading {}...'.format(filename))
    start = time.time()

    model = EastModel(config)
    model.load_state_dict(torch.load(filename))
    print('elapsed {} sec'.format(time.time() - start))
