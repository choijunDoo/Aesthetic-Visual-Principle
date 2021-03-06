"""
file - main.py
Main script to train the aesthetic model on the AVA dataset.
Copyright (C) Yunxiao Shi 2017 - 2021
NIMA is released under the MIT license. See LICENSE for the fill license text.
"""

import argparse
import os

import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.autograd as autograd
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torchvision.models as models

from torch.utils.tensorboard import SummaryWriter

from load_dataset.dataset import AVADataset

from model.model import *
from losses import emd_loss


def main(config):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter()

    train_transform = transforms.Compose([
        transforms.Scale(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225])])

    val_transform = transforms.Compose([
        transforms.Scale(256),
        transforms.RandomCrop(224),
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225])])

    model = PANet("GCN_Element", device)
    model = model.to(device)

    optimizer = optim.SGD(model.parameters(), momentum=0.9, lr= config.conv_base_lr)
    param_num = 0
    for param in model.parameters():
        if param.requires_grad:
            param_num += param.numel()
    print('Trainable params: %.2f million' % (param_num / 1e6))
    print(device)

    if config.train:
        trainset = AVADataset(csv_file=config.train_csv_file, root_dir=config.img_path, transform=train_transform)
        valset = AVADataset(csv_file=config.val_csv_file, root_dir=config.img_path, transform=val_transform)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=config.train_batch_size,
            shuffle=True, num_workers=config.num_workers)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=config.val_batch_size,
            shuffle=False, num_workers=config.num_workers)
        # for early stopping
        count = 0
        init_val_loss = float('inf')
        train_losses = []
        val_losses = []

        for epoch in range(config.warm_start_epoch, config.epochs):
            batch_losses = []
            for i, data in enumerate(train_loader):
                images = data['image'].to(device)
                labels = data['annotations'].to(device).float()
                outputs = model(images, len(data['image']))
                outputs = outputs.view(-1, 10, 1)

                optimizer.zero_grad()

                loss = emd_loss(labels, outputs)
                batch_losses.append(loss.item())

                loss.backward()

                optimizer.step()

                print('Epoch: %d/%d | Step: %d/%d | Training EMD loss: %.4f' % (epoch + 1, config.epochs, i + 1, len(trainset) // config.train_batch_size + 1, loss.data[0]))
                writer.add_scalar('batch train loss', loss.data[0], i + epoch * (len(trainset) // config.train_batch_size + 1))

            avg_loss = sum(batch_losses) / (len(trainset) // config.train_batch_size + 1)
            train_losses.append(avg_loss)
            print('Epoch %d mean training EMD loss: %.4f' % (epoch + 1, avg_loss))

            # exponetial learning rate decay
            if config.decay:
                if (epoch + 1) % 10 == 0:
                    conv_base_lr = conv_base_lr * config.lr_decay_rate ** ((epoch + 1) / config.lr_decay_freq)
                    optimizer = optim.SGD(model.parameters(), momentum=0.9, lr = conv_base_lr)

            # do validation after each epoch
            batch_val_losses = []
            for data in val_loader:
                images = data['image'].to(device)
                labels = data['annotations'].to(device).float()
                with torch.no_grad():
                    outputs = model(images, len(data['image']))
                outputs = outputs.view(-1, 10, 1)
                val_loss = emd_loss(labels, outputs)
                batch_val_losses.append(val_loss.item())
            avg_val_loss = sum(batch_val_losses) / (len(valset) // config.val_batch_size + 1)
            val_losses.append(avg_val_loss)
            print('Epoch %d completed. Mean EMD loss on val set: %.4f.' % (epoch + 1, avg_val_loss))
            writer.add_scalars('epoch losses', {'epoch train loss': avg_loss, 'epoch val loss': avg_val_loss}, epoch + 1)

            # Use early stopping to monitor training
            if avg_val_loss < init_val_loss:
                init_val_loss = avg_val_loss
                # save model weights if val loss decreases
                print('Saving model...')
                if not os.path.exists(config.ckpt_path):
                    os.makedirs(config.ckpt_path)
                torch.save(model.state_dict(), os.path.join(config.ckpt_path, 'Proposed-Non-epoch-%d.pth' % (epoch + 1)))
                print('Done.\n')
                # reset count
                count = 0
            elif avg_val_loss >= init_val_loss:
                count += 1
                if count == config.early_stopping_patience:
                    print('Val EMD loss has not decreased in %d epochs. Training terminated.' % config.early_stopping_patience)
                    break

        print('Training completed.')

        '''
        # use tensorboard to log statistics instead
        if config.save_fig:
            # plot train and val loss
            epochs = range(1, epoch + 2)
            plt.plot(epochs, train_losses, 'b-', label='train loss')
            plt.plot(epochs, val_losses, 'g-', label='val loss')
            plt.title('EMD loss')
            plt.legend()
            plt.savefig('./loss.png')
        '''

    if config.test:
        model.eval()
        # compute mean score
        testset = AVADataset(csv_file=config.test_csv_file, root_dir=config.img_path, transform=val_transform)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=config.test_batch_size, shuffle=False, num_workers=config.num_workers)

        mean_preds = []
        std_preds = []
        preds = []
        gt = []

        for data in test_loader:
            image = data['image'].to(device)
            labels = data['annotations'].to(device).float()
            output = model(image)
            output = output.view(10, 1)
            predicted_mean, predicted_std, gt_mean = 0.0, 0.0, 0.0
            for i, elem in enumerate(output, 1):
                predicted_mean += i * elem

            for k, elem in enumerate(labels, 1):
                gt_mean += k * elem

            for j, elem in enumerate(output, 1):
                predicted_std += elem * (j - predicted_mean) ** 2

            predicted_std = predicted_std ** 0.5
            mean_preds.append(predicted_mean)
            std_preds.append(predicted_std)

            if predicted_mean >= 0.5:
                preds.append(1)
            else:
                preds.append(0)

            if gt_mean >= 0.5:
                gt.append(1)
            else:
                gt.append(0)
        # Do what you want with predicted and std...


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # input parameters
    parser.add_argument('--img_path', type=str, default='./dataset/images/images/')
    parser.add_argument('--train_csv_file', type=str, default='./dataset/train.csv')
    parser.add_argument('--val_csv_file', type=str, default='./dataset/val.csv')
    parser.add_argument('--test_csv_file', type=str, default='./dataset/test.csv')

    # training parameters
    parser.add_argument('--train',action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--decay', action='store_true')
    parser.add_argument('--conv_base_lr', type=float, default=5e-3)
    parser.add_argument('--dense_lr', type=float, default=5e-4)
    parser.add_argument('--lr_decay_rate', type=float, default=0.95)
    parser.add_argument('--lr_decay_freq', type=int, default=10)
    parser.add_argument('--train_batch_size', type=int, default=128)
    parser.add_argument('--val_batch_size', type=int, default=128)
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=100)

    # misc
    parser.add_argument('--ckpt_path', type=str, default='./save2')
    parser.add_argument('--multi_gpu', action='store_true')
    parser.add_argument('--gpu_ids', type=list, default=None)
    parser.add_argument('--warm_start', action='store_true')
    parser.add_argument('--warm_start_epoch', type=int, default=0)
    parser.add_argument('--early_stopping_patience', type=int, default=10)
    parser.add_argument('--save_fig', action='store_true')

    config = parser.parse_args()

    main(config)