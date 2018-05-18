import argparse
import os
import logging
import sys

import torch
from torch.utils.data import DataLoader, ConcatDataset

from vision.utils.misc import str2bool, Timer
from vision.ssd.ssd import MatchPrior
from vision.ssd.vgg_ssd import create_vgg_ssd
from vision.datasets.voc_dataset import VOCDataset, class_names
from vision.nn.multibox_loss import MultiboxLoss
from vision.ssd.config import vgg_ssd_config as config
from vision.ssd.data_preprocessing import TrainAugmentation, TestTransform


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')

parser.add_argument('--datasets', nargs='+', help='Dataset directory path')
parser.add_argument('--validation_dataset', help='Dataset directory path')

# Params for SGD
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')

# Params for loading pretrained basenet or checkpoints.
parser.add_argument('--base_net', default='models/vgg16_reducedfc.pth',
                    help='Pretrained base model')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')

# Train params
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--num_epochs', default=10, type=int,
                    help='the number epochs')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--validation_epochs', default=5, type=int,
                    help='the number epochs')
parser.add_argument('--debug_steps', default=100, type=int,
                    help='Set the debug log output frequency.')
parser.add_argument('--use_cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')

parser.add_argument('--checkpoint_folder', default='models/',
                    help='Directory for saving checkpoint models')


args = parser.parse_args()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")


def train(loader, net, criterion, optimizer, device, debug_steps=100):
    net.train(True)
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    for i, data in enumerate(loader):
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        confidence, locations = net(images)
        regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)  # TODO CHANGE BOXES
        loss = regression_loss + classification_loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
        if i and i % debug_steps == 0:
            logging.info(
                "Step: {}, Average Loss: {:.4f}, Average Regression Loss {:.4f}, Average Classification Loss: {:.4f}".format(
                    i,
                    running_loss / debug_steps,
                    running_regression_loss / debug_steps,
                    running_classification_loss / debug_steps
                ))
            running_loss = 0.0
            running_regression_loss = 0.0
            running_classification_loss = 0.0


def test(loader, net, criterion, device):
    net.eval()
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    num = 0
    for _, data in enumerate(loader):
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)
        num += 1

        with torch.no_grad():
            confidence, locations = net(images)
            regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)
            loss = regression_loss + classification_loss

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
    return running_loss / num, running_regression_loss / num, running_classification_loss / num


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logging.info("Build network.")
    logging.info(args)
    timer = Timer()
    net = create_vgg_ssd(len(class_names))
    timer.start("Load Model")
    if args.resume:
        net.load(args.resume)
    elif args.base_net:
        net.init_from_base_net(args.base_net)
    logging.info(f'It took {timer.end("Load Model")} seconds to load the model.')

    if args.use_cuda and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    net = net.to(DEVICE)

    criterion = MultiboxLoss(config.priors, iou_threshold=0.5, neg_pos_ratio=3,
                             center_variance=0.1, size_variance=0.2, device=DEVICE)
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[19, 95, 114], gamma=0.1)

    train_transform = TrainAugmentation(config.image_size, config.image_mean)
    target_transform = MatchPrior(config.priors, config.center_variance,
                                  config.size_variance, 0.5)

    test_transform = TestTransform(config.image_size, config.image_mean)

    datasets = []
    for dataset_path in args.datasets:
        dataset = VOCDataset(dataset_path, transform=train_transform,
                               target_transform=target_transform)
        datasets.append(dataset)
    train_dataset = ConcatDataset(datasets)
    logging.info("Train dataset size: {}".format(len(train_dataset)))
    train_loader = DataLoader(train_dataset, args.batch_size,
                              num_workers=args.num_workers,
                              shuffle=False)

    val_dataset = VOCDataset(args.validation_dataset, transform=test_transform,
                             target_transform=target_transform, is_test=True)
    logging.info("validation dataset size: {}".format(len(val_dataset)))

    val_loader = DataLoader(val_dataset, args.batch_size,
                            num_workers=args.num_workers,
                            shuffle=False)
    logging.info("Start training.")
    min_loss = 100000.0
    for epoch in range(args.num_epochs):
        scheduler.step()
        train(train_loader, net, criterion, optimizer,
              device=DEVICE, debug_steps=args.debug_steps)
        logging.info("save model.")
        torch.save(
            net.state_dict(),
            os.path.join(
                args.checkpoint_folder, "vgg-ssd-epoch-{}-loss-{:.4f}.pth".format(epoch, epoch)
            )
        )
        if epoch % args.validation_epochs == 0:
            val_loss, val_regression_loss, val_classification_loss = test(val_loader, net, criterion, DEVICE)
            logging.info("Epoch: {}, Validation Loss: {:.4f}, Validation Regression Loss {:.4f}, Validation Classification Loss: {:.4f}".format(
                epoch,
                val_loss,
                val_regression_loss,
                val_classification_loss
            ))
            if val_loss < min_loss:
                min_loss = val_loss
                torch.save(
                    net.state_dict(),
                    os.path.join(
                        args.checkpoint_folder, "VGG-SSD-Epoch-{}-Loss-{:.4f}".format(epoch, val_loss)
                    )
                )
