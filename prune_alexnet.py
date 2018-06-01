import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
import argparse
import logging
import sys
from tensorboardX import SummaryWriter

from vision.prunning.prunner import ModelPrunner
from vision.utils.misc import str2bool
from vision.nn.alexnet import alexnet


parser = argparse.ArgumentParser(description='Demonstration of Pruning AlexNet')

parser.add_argument("--train", dest="train", action="store_true")
parser.add_argument("--prune_conv", dest="prune_conv", action="store_true")
parser.add_argument("--prune_linear", dest="prune_linear", action="store_true")
parser.add_argument("--trained_model", type=str)
parser.add_argument('--dataset', type=str, help='Dataset directory path')
parser.add_argument('--validation_dataset', help='Dataset directory path')
parser.add_argument('--batch_size', default=12, type=int,
                    help='Batch size for training')
parser.add_argument('--num_epochs', default=25, type=int,
                    help='number of batches to train')
parser.add_argument('--num_recovery_batches', default=2, type=int,
                    help='number of batches to train to recover the network')
parser.add_argument('--recovery_learning_rate', default=1e-4, type=float,
                    help='learning rate to recover the network')
parser.add_argument('--recovery_batch_size', default=32, type=int,
                    help='Batch size for training')

# Params for SGD
parser.add_argument('--learning_rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')

# Params for Pruning
parser.add_argument('--prune_conv_num', default=1, type=int,
                    help='the number of conv filters you want to prune in very iteration.')
parser.add_argument('--prune_linear_num', default=2, type=int,
                    help='the number of linear filters you want to prune in very iteration.')
parser.add_argument('--window', default=10, type=int,
                    help='Window size for tracking training accuracy.')

parser.add_argument('--use_cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')


args = parser.parse_args()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")
cpu_device = torch.device("cpu")


if args.use_cuda and torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


def train_epoch(net, data_iter, num_epochs=1, optimizer=None):
    net = net.to(DEVICE)
    net.train()
    criterion = nn.CrossEntropyLoss()

    num = 0
    for i in range(num_epochs):
        inputs, labels = next(data_iter)
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)
        if optimizer:
            optimizer.zero_grad()

        outputs = net(inputs)

        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        loss.backward()
        if optimizer:
            optimizer.step()
        train_loss = loss.item() * inputs.size(0)
        train_accuracy = torch.sum(preds == labels.data).item()
        num += inputs.size(0)
    train_loss /= num
    train_accuracy /= num
    logging.info('Train Epoch Loss:{:.4f}, Accuracy:{:.4f}'.format(train_loss, train_accuracy))
    return train_loss, train_accuracy


def train(net, train_loader, val_loader, num_epochs, learning_rate, save_model=True):
    net = net.to(DEVICE)
    net.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate,
                          momentum=args.momentum, weight_decay=args.weight_decay)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    for i in range(num_epochs):
        net.train()
        exp_lr_scheduler.step()
        num = 0
        running_loss = 0.0
        running_corrects = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(inputs)

            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()
            num += inputs.size(0)

        logging.info('Epoch: {}, Training Loss:{:.4f}, Training Accuracy:{:.4f}'.format(i, running_loss/num, running_corrects/num))
        val_loss, val_accuracy = eval(net, val_loader)
        logging.info('Epoch: {}, Val Loss:{:.4f}, Val Accuracy:{:.4f}'.format(i, val_loss, val_accuracy))
        if save_model:
            torch.save(net.state_dict(), "models/ant-alexnet-epoch-{}-{:.4f}.pth".format(i, val_accuracy))
    return val_loss, val_accuracy


def eval(net, loader):
    net.eval()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    running_corrects = 0
    num = 0
    for inputs, labels in loader:
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)
        with torch.set_grad_enabled(False):
            outputs = net(inputs)

            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data).item()
        num += inputs.size(0)
    running_loss /= num
    running_corrects = running_corrects / num
    return running_loss, running_corrects


def make_prunner_loader(dataset):
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.recovery_batch_size, shuffle=True, num_workers=1)
    while True:
        for inputs, labels in loader:
            yield inputs, labels

if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    net = alexnet(True)
    net.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 2),
        )

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(args.dataset, train_transform)
    val_dataset = datasets.ImageFolder(args.validation_dataset, val_transform)
    logging.info(f"Training dataset size: {len(train_dataset)}.")
    logging.info(f"Validation Dataset size: {len(val_dataset)}.")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False, num_workers=1)
    writer = SummaryWriter()
    if args.train:
        logging.info("Start training.")
        train(net, train_loader, val_loader, args.num_epochs, args.learning_rate)
    elif args.prune_conv or args.prune_linear:
        net.load_state_dict(torch.load(args.trained_model))
        prunner_data_iter = iter(make_prunner_loader(train_dataset))
        prunner = ModelPrunner(net, lambda model: train_epoch(model, prunner_data_iter),
                               ignored_paths=[('classifier', '6')])  # do not prune the last layer.
        num_filters = prunner.book.num_of_conv2d_filters()
        logging.info(f"Number of Conv2d filters: {num_filters}")

        num_linear_filters = prunner.book.num_of_linear_filters()
        logging.info(f"Number of Linear filters: {num_linear_filters}")
        if args.prune_conv:
            prune_num = prunner.book.num_of_conv2d_filters() - 5 * (prunner.book.num_of_conv2d_modules())
        else:
            prune_num = prunner.book.num_of_linear_filters() - 5 * (prunner.book.num_of_linear_modules())
        logging.info(f"Number of Layers to Prune: {prune_num}")
        i = 0
        iteration = 0
        train_data_iter = iter(make_prunner_loader(train_dataset))
        optimizer = optim.SGD(net.parameters(), lr=args.recovery_learning_rate,
                              momentum=args.momentum, weight_decay=args.weight_decay)
        while i < prune_num:
            if args.prune_conv:
                prunner.prune_conv_layers(args.prune_conv_num)
                i += args.prune_conv_num
            else:
                _, accuracy_gain = prunner.prune_linear_layers(args.prune_linear_num)
                i += args.prune_linear_num
            if iteration % 10 == 0:
                val_loss, val_accuracy = eval(prunner.model, val_loader)
                logging.info(f"Prune: {i}/{prune_num}, After Pruning Evaluation Accuracy:{val_accuracy:.4f}.")
            val_loss, val_accuracy = train_epoch(prunner.model, train_data_iter, args.num_recovery_batches, optimizer)
            for name, param in net.named_parameters():
                writer.add_histogram(name, param.clone().cpu().data.numpy(), 10)
            if iteration % 10 == 0:
                dummy_input = torch.rand(1, 3, 224, 224)
                writer.add_graph(net, dummy_input)
                val_loss, val_accuracy = eval(prunner.model, val_loader)
                logging.info(f"Prune: {i}/{prune_num}, After Recovery Evaluation Accuracy:{val_accuracy:.4f}.")
                logging.info(f"Prune: {i}/{prune_num}, Iteration: {iteration}, Save model.")
                with open(f"models/alexnet-pruned-{i}.txt", "w") as f:
                    print(prunner.model, file=f)
                torch.save(prunner.model.state_dict(), f"models/prunned-alexnet-{i}-{prune_num}-{val_accuracy:.4f}.pth")
            iteration += 1
    else:
        logging.fatal("You should specify --prune_conv, --prune_linear or --train.")

    writer.close()
