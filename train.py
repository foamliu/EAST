import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import nn

from config import device, grad_clip, print_freq, num_workers
from data_gen import EastDataset, collate_fn
from eval import predict
from loss import LossFunc
from models import EastModel
from script import default_evaluation_params, evaluate_method
from utils import parse_args, save_checkpoint, AverageMeter, clip_gradient, get_logger, get_learning_rate, \
    adjust_learning_rate


def train_net(args):
    torch.manual_seed(7)
    np.random.seed(7)
    checkpoint = args.checkpoint
    start_epoch = 0
    best_hmean = 0
    writer = SummaryWriter()
    epochs_since_improvement = 0
    decays_since_improvement = 0

    # Initialize / load checkpoint
    if checkpoint is None:
        model = EastModel(args)
        model = nn.DataParallel(model)

        if args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.mom,
                                        weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    logger = get_logger()

    # Move to GPU, if available
    model = model.to(device)

    # Loss function
    criterion = LossFunc()

    # Custom dataloaders
    train_dataset = EastDataset('train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
    # test_dataset = EastDataset('test')
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size_per_gpu * args.num_gpus,
    #                                           shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

    # Epochs
    for epoch in range(start_epoch, args.end_epoch):
        # Decay learning rate if there is no improvement for 2 consecutive epochs, and terminate training after 10
        if args.optimizer == 'sgd' and epochs_since_improvement == 10:
            break
        if args.optimizer == 'sgd' and epochs_since_improvement > 0 and epochs_since_improvement % 2 == 0:
            checkpoint = 'BEST_checkpoint.tar'
            checkpoint = torch.load(checkpoint)
            model = checkpoint['model']
            optimizer = checkpoint['optimizer']
            decays_since_improvement += 1
            print("\nDecays since last improvement: %d\n" % (decays_since_improvement,))
            adjust_learning_rate(optimizer, 0.6 ** decays_since_improvement)

        # One epoch's training
        train_loss = train(train_loader=train_loader,
                           model=model,
                           criterion=criterion,
                           optimizer=optimizer,
                           epoch=epoch,
                           logger=logger)
        effective_lr = get_learning_rate(optimizer)
        print('\nCurrent effective learning rate: {}\n'.format(effective_lr))

        writer.add_scalar('Train_Loss', train_loss, epoch)
        writer.add_scalar('Learning_Rate', effective_lr, epoch)

        # One epoch's validation
        test_hmean = compute_hmean(model=model)
        writer.add_scalar('hmean', test_hmean, epoch)

        # Check if there was an improvement
        is_best = test_hmean > best_hmean
        best_hmean = max(test_hmean, best_hmean)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0
            decays_since_improvement = 0

        # Save checkpoint
        save_checkpoint(epoch, epochs_since_improvement, model, optimizer, best_hmean, is_best)


def train(train_loader, model, criterion, optimizer, epoch, logger):
    model.train()  # train mode (dropout and batchnorm is used)

    losses = AverageMeter()

    # Batches
    for i, (img, score_map, geo_map, training_mask) in enumerate(train_loader):
        # Move to GPU, if available
        img = img.to(device)
        score_map = score_map.to(device)
        geo_map = geo_map.to(device)
        training_mask = training_mask.to(device)

        # Forward prop.
        f_score, f_geometry = model(img)

        # Calculate loss
        loss = criterion(score_map, f_score, geo_map, f_geometry, training_mask)

        # Back prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        clip_gradient(optimizer, grad_clip)

        # Update weights
        optimizer.step()

        # Keep track of metrics
        losses.update(loss.item(), img.size(0))

        # Print status
        if i % print_freq == 0:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(epoch, i, len(train_loader), loss=losses))

    return losses.avg


# def test(test_loader, model, criterion, logger):
#     model.eval()  # train mode (dropout and batchnorm is used)
#
#     losses = AverageMeter()
#
#     # Batches
#     for img, score_map, geo_map, training_mask in tqdm(test_loader):
#         # Move to GPU, if available
#         img = img.to(device)
#         score_map = score_map.to(device)
#         geo_map = geo_map.to(device)
#         training_mask = training_mask.to(device)
#
#         # Forward prop.
#         f_score, f_geometry = model(img)
#
#         # Calculate loss
#         loss = criterion(score_map, f_score, geo_map, f_geometry, training_mask)
#
#         # Keep track of metrics
#         losses.update(loss.item(), img.size(0))
#
#     # Print status
#     logger.info('TEST Loss {loss.val:.4f} ({loss.avg:.4f})\n'.format(loss=losses))
#
#     return losses.avg

def compute_hmean(model):
    model.eval()
    predict(model, 0)

    evalParams = default_evaluation_params()
    gtFilePath = 'gt.zip'
    submFilePath = 'data/result/epoch_0_gt'
    resDict = {'calculated': True, 'Message': '', 'method': '{}', 'per_sample': '{}'}
    evalData = evaluate_method(gtFilePath, submFilePath, evalParams)
    resDict.update(evalData)
    ret = resDict['method']
    print(ret)

    return ret['hmean']


def main():
    global args
    args = parse_args()
    train_net(args)


if __name__ == '__main__':
    main()
