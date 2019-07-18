import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import nn

from config import device, grad_clip, print_freq, num_workers
from data_gen import EastDataset
from models import resnet18, resnet34, resnet50, resnet101, resnet152
from utils import parse_args, save_checkpoint, AverageMeter, clip_gradient, accuracy, get_logger, adjust_learning_rate, \
    get_learning_rate


def train_net(args):
    torch.manual_seed(7)
    np.random.seed(7)
    checkpoint = args.checkpoint
    start_epoch = 0
    best_loss = float("inf")
    writer = SummaryWriter()
    epochs_since_improvement = 0

    # Initialize / load checkpoint
    if checkpoint is None:
        if args.network == 'r18':
            model = resnet18(args)
        elif args.network == 'r34':
            model = resnet34(args)
        elif args.network == 'r50':
            model = resnet50(args)
        elif args.network == 'r101':
            model = resnet101(args)
        elif args.network == 'r152':
            model = resnet152(args)
        else:
            model = resnet50(args)
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
    criterion = nn.CrossEntropyLoss().to(device)

    # Custom dataloaders
    train_dataset = EastDataset('train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=num_workers)
    valid_dataset = EastDataset('valid')
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,
                                               num_workers=num_workers)

    # Epochs
    for epoch in range(start_epoch, args.end_epoch):
        # Decay learning rate if there is no improvement for 2 consecutive epochs, and terminate training after 10
        if epochs_since_improvement == 10:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 2 == 0:
            checkpoint = 'BEST_checkpoint.tar'
            checkpoint = torch.load(checkpoint)
            model = checkpoint['model']
            optimizer = checkpoint['optimizer']

            adjust_learning_rate(optimizer, 0.5)

        # One epoch's training
        train_loss, train_top1_accs = train(train_loader=train_loader,
                                            model=model,
                                            criterion=criterion,
                                            optimizer=optimizer,
                                            epoch=epoch,
                                            logger=logger)
        effective_lr = get_learning_rate(optimizer)
        print('\nCurrent effective learning rate: {}\n'.format(effective_lr))

        writer.add_scalar('Train_Loss', train_loss, epoch)
        writer.add_scalar('Train_Top1_Accuracy', train_top1_accs, epoch)
        writer.add_scalar('Learning_Rate', effective_lr, epoch)

        # One epoch's validation
        valid_loss, valid_top1_accs = valid(valid_loader=valid_loader,
                                            model=model,
                                            criterion=criterion,
                                            epoch=epoch,
                                            logger=logger)
        writer.add_scalar('Valid_Loss', valid_loss, epoch)
        writer.add_scalar('Valid_Top1_Accuracy', valid_top1_accs, epoch)

        # Check if there was an improvement
        is_best = valid_loss < best_loss
        best_loss = min(valid_loss, best_loss)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(epoch, epochs_since_improvement, model, optimizer, best_loss, is_best)


def train(train_loader, model, criterion, optimizer, epoch, logger):
    model.train()  # train mode (dropout and batchnorm is used)

    losses = AverageMeter()
    top1_accs = AverageMeter()

    # Batches
    for i, (img, label) in enumerate(train_loader):
        # Move to GPU, if available
        img = img.to(device)
        label = label.to(device)  # [N, 1]

        # Forward prop.
        output = model(img)  # embedding => [N, 512]

        # Calculate loss
        loss = criterion(output, label)

        # Back prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        clip_gradient(optimizer, grad_clip)

        # Update weights
        optimizer.step()

        # Keep track of metrics
        losses.update(loss.item())
        top1_accuracy = accuracy(output, label, 15)
        top1_accs.update(top1_accuracy)

        # Print status
        if i % print_freq == 0:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Top1 Accuracy {top1_accs.val:.3f} ({top1_accs.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                                         loss=losses,
                                                                                         top1_accs=top1_accs))

    return losses.avg, top1_accs.avg


def valid(valid_loader, model, criterion, epoch, logger):
    model.eval()  # train mode (dropout and batchnorm is used)

    losses = AverageMeter()
    top1_accs = AverageMeter()

    # Batches
    for i, (img, label) in enumerate(valid_loader):
        # Move to GPU, if available
        img = img.to(device)
        label = label.to(device)  # [N, 1]

        # Forward prop.
        output = model(img)  # embedding => [N, 512]

        # Calculate loss
        loss = criterion(output, label)

        # Keep track of metrics
        losses.update(loss.item())
        top1_accuracy = accuracy(output, label, 1)
        top1_accs.update(top1_accuracy)

        # Print status
        if i % print_freq == 0:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Top1 Accuracy {top1_accs.val:.3f} ({top1_accs.avg:.3f})'.format(epoch, i, len(valid_loader),
                                                                                         loss=losses,
                                                                                         top1_accs=top1_accs))

    return losses.avg, top1_accs.avg


def main():
    global args
    args = parse_args()
    train_net(args)


if __name__ == '__main__':
    main()
