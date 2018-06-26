# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 12:16:31 2017

@author: Biagio Brattoli
"""
import argparse
import os
import sys
from time import time

import numpy as np

sys.path.append('Utils')
from logger import Logger

import torch
import torch.nn as nn
from torch.autograd import Variable

sys.path.append('dataset')
from JigsawNetwork import Network

from TrainingUtils import adjust_learning_rate, compute_accuracy
from dicom_jigsaw_loader import DicomDataset

parser = argparse.ArgumentParser(description='Train JigsawPuzzleSolver on Imagenet')
parser.add_argument('data', type=str, nargs='+', help='Path(s) to Dicom folder(s)')
parser.add_argument('--model', default=None, type=str, help='Path to pretrained model')
parser.add_argument('--classes', default=1000, type=int, help='Number of permutation to use')
parser.add_argument('--gpu', default=1, type=int, help='gpu id')
parser.add_argument('--epochs', default=500, type=int, help='number of total epochs for training')
parser.add_argument('--fast', default=False, action='store_true', help='fast assumes same input files')
parser.add_argument('--iter_start', default=0, type=int, help='Starting iteration count')
parser.add_argument('--batch', default=256, type=int, help='batch size')
parser.add_argument('--checkpoint', default='checkpoints/', type=str, help='checkpoint folder')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate for SGD optimizer')
parser.add_argument('--cores', default=0, type=int, help='number of CPU cores for loading')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set, No training')
parser.add_argument('--blocks', default=False, action='store_true', help='show blocks in images')
args = parser.parse_args()


def main():
    if args.gpu is not None:
        print(('Using GPU %d' % args.gpu))
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    else:
        print('CPU mode')

    print('Process number: %d' % (os.getpid()))

    trainpath = args.data
    for p in trainpath:
        assert os.path.exists(p)

    # The first instance of DicomDataset currently splits and stores exams for train/val
    train_data = DicomDataset(trainpath, classes=args.classes, fast=args.fast, show_blocks=args.blocks)
    train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                               batch_size=args.batch,
                                               shuffle=True,
                                               num_workers=args.cores)

    val_data = DicomDataset(trainpath, classes=args.classes, train=False)
    val_loader = torch.utils.data.DataLoader(dataset=val_data,
                                             batch_size=args.batch,
                                             shuffle=True,
                                             num_workers=args.cores)
    iter_per_epoch = len(train_data) / args.batch
    print('Images: train %d, validation %d' % (len(train_data), len(val_data)))

    # Network initialize
    net = Network(args.classes)
    if args.gpu is not None:
        net.cuda()

    ############## Load from checkpoint if exists, otherwise from model ###############
    if os.path.exists(args.checkpoint):
        files = [f for f in os.listdir(args.checkpoint) if 'pth' in f]
        if len(files) > 0:
            files.sort()
            # print files
            ckp = files[-1]
            net.load_state_dict(torch.load(args.checkpoint + '/' + ckp))
            args.iter_start = int(ckp.split(".")[-3].split("_")[-1])
            print('Starting from: ', ckp)
        else:
            if args.model is not None:
                net.load(args.model)
    else:
        if args.model is not None:
            net.load(args.model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    logger = Logger(args.checkpoint + '/train')
    logger_test = Logger(args.checkpoint + '/test')

    ############## TESTING ###############
    if args.evaluate:
        test(net, criterion, None, val_loader, 0)
        return

    ############## TRAINING ###############
    print(('Start training: lr %f, batch size %d, classes %d' % (args.lr, args.batch, args.classes)))
    print(('Checkpoint: ' + args.checkpoint))

    # Train the Model
    batch_time, net_time = [], []
    steps = args.iter_start
    for epoch in range(int(args.iter_start / iter_per_epoch), args.epochs):
        if epoch % 10 == 0 and epoch > 0:
            test(net, criterion, logger_test, val_loader, steps)
        lr = adjust_learning_rate(optimizer, epoch, init_lr=args.lr, step=50, decay=0.1)

        end = time()
        for i, (images, labels, original) in enumerate(train_loader):
            batch_time.append(time() - end)
            if len(batch_time) > 100:
                del batch_time[0]

            images = Variable(images).float()
            labels = Variable(labels)
            if args.gpu is not None:
                images = images.cuda()
                labels = labels.cuda()

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            t = time()
            images.unsqueeze_(2)  # put this in the dataset
            outputs = net(images)
            net_time.append(time() - t)
            net_time = net_time[-100:]

            prec1, prec5 = compute_accuracy(outputs.cpu().data, labels.cpu().data, topk=(1, 5))
            acc = prec1.item()

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss = float(loss.cpu().data.numpy())

            if steps % 20 == 0:
                print(
                    ('[%2d/%2d] %5d) [batch load % 2.3fsec, net %1.2fsec], LR %.5f, Loss: % 1.3f, Acc % 2.2f%%' % (
                        epoch + 1, args.epochs, steps,
                        np.mean(batch_time), np.mean(net_time),
                        lr, loss, acc)))

            if steps % 20 == 0:
                logger.scalar_summary('accuracy', acc, steps)
                logger.scalar_summary('loss', loss, steps)

                original = [im[0] for im in original]
                imgs = np.zeros([9, 75, 75])

                def normalize_img(im):
                    intensity_range = im.max() - im.min()
                    if intensity_range == 0.0:
                        return np.zeros_like(im)
                    return (im - im.min()) / intensity_range

                for ti, img in enumerate(original):
                    img = img.numpy()
                    imgs[ti] = np.stack([normalize_img(im) for im in img], axis=1)

                logger.image_summary('input', imgs, steps)

            steps += 1

            if steps % 1000 == 0:
                filename = '%s/jps_%03i_%06d.pth.tar' % (args.checkpoint, epoch, steps)
                net.save(filename)
                print('Saved: ' + args.checkpoint)

            end = time()

        if os.path.exists(args.checkpoint + '/stop.txt'):
            # break without using CTRL+C
            break


def test(net, criterion, logger, val_loader, steps):
    print('Evaluating network.......')
    accuracy = []
    net.eval()
    for i, (images, labels, _) in enumerate(val_loader):
        images = Variable(images).float()
        if args.gpu is not None:
            images = images.cuda()

        # Forward + Backward + Optimize
        images.unsqueeze_(2)  # put this in the dataset
        outputs = net(images)
        outputs = outputs.cpu().data

        prec1, prec5 = compute_accuracy(outputs, labels, topk=(1, 5))
        accuracy.append(prec1.item())

    if logger is not None:
        logger.scalar_summary('accuracy', np.mean(accuracy), steps)
    print('TESTING: %d), Accuracy %.5f%%' % (steps, np.mean(accuracy)))
    net.train()


if __name__ == "__main__":
    main()
