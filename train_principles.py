import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.autograd as autograd
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torchvision.models as models

from torch.utils.tensorboard import SummaryWriter

from load_dataset.dataset import PDataset
from losses import focal_binary_cross_entropy, FocalLoss

from model.model import *
from evalution_metrics import AverageMeter, accuracy, pred_acc

import warnings
warnings.filterwarnings('ignore')

def train(train_loader, val_loader, model, loss_fn, lr = 1e-4, epochs = 50, decay = 'store_true', lr_decay_rate = 0.95, lr_decay_freq = 10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    writer = SummaryWriter()

    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    param_num = 0
    for param in model.parameters():
        if param.requires_grad:
            param_num += param.numel()
    print('Trainable params: %.2f million' % (param_num / 1e6))

    train_loss_arr = []
    train_acc_arr = []
    train_prec_arr = [] # multi_label

    val_loss_arr = []
    val_acc_arr = []
    val_prec_arr = [] #  multi-label

    for epoch in range(0, epochs):

        model.train()

        losses = AverageMeter()
        top1 = AverageMeter()
        precisions = AverageMeter()

        batch_losses = []
        for i, data in enumerate(train_loader):
            images = data['image'].to(device)
            labels = data['information'].long().to(device)  ## (batch 128, 8)
            outputs = model(images)  # (batch, 128, 8)
            _, labels = torch.max(labels, 1)
            #         outputs = outputs.view(-1, 10, 1)
            #         outputs = torch.sigmoid(outputs)
            optimizer.zero_grad()

            outputs.float()
            #         labels = labels.float()

            loss = loss_fn(outputs, labels)  ##

            loss.float()
            #         batch_losses.append(loss.item())

            prec1 = accuracy(outputs.data, labels)  ##
            prec1 = prec1[0]
            #         precision = precision_score(np.round(outputs.detach().cpu().numpy()), labels.detach().cpu().numpy(), average = 'samples')

            losses.update(loss.item(), len(data['img_id']))
            top1.update(prec1, len(data['img_id']))
            #         precisions.update(precision, len(data['img_id']))

            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(epoch, i, len(train_loader), loss=losses))
        #         writer.add_scalar('batch train loss', loss.data, i + epoch * (len(trainset) // 128 + 1))

        #     avg_loss = sum(batch_losses) / (len(trainset) // 128 + 1)
        #     train_losses.append(avg_loss)
        #     print('Epoch %d mean training loss: %.4f' % (epoch + 1, avg_loss))

        train_loss_arr.append(losses.avg)
        train_acc_arr.append(top1.avg)
        #     train_prec_arr.append(precisions.avg)
        print("train result: Loss: %.4f, Acc: %.4f" % (losses.avg, top1.avg))

        # exponetial learning rate decay
        if decay:
            if (epoch + 1) % 10 == 0:
                conv_base_lr = conv_base_lr * lr_decay_rate ** ((epoch + 1) / lr_decay_freq)
                #             dense_lr = dense_lr * lr_decay_rate ** ((epoch + 1) / lr_decay_freq)
                #             optimizer = optim.SGD([
                #                 {'params': model.features.parameters(), 'lr': conv_base_lr},
                #                 {'params': model.classifier.parameters(), 'lr': dense_lr}],
                #                 momentum=0.9
                #             )
                optimizer = optim.SGD(model.parameters(), lr=conv_base_lr, momentum=0.9)

        model.eval()
        with torch.no_grad():
            val_loss_sum = 0
            val_acc_sum = 0

            losses = AverageMeter()
            top1 = AverageMeter()
            #         precisions = AverageMeter()

            for i, data in enumerate(val_loader):
                images = data['image'].to(device)
                labels = data['information'].long().to(device)  ## (batch 128, 8)
                outputs = model(images)  # (batch, 128, 8)
                _, labels = torch.max(labels, 1)
                #             labels = labels.float()

                #             outputs = torch.sigmoid(outputs)
                outputs.float()

                loss = loss_fn(outputs, labels)

                outputs.float()
                loss.float()

                prec1 = accuracy(outputs.data, labels)
                prec1 = prec1[0]
                #             precision = precision_score(np.round(outputs.detach().cpu().numpy()), labels.detach().cpu().numpy(), average = 'samples')

                losses.update(loss.item(), len(data['img_id']))
                top1.update(prec1, len(data['img_id']))
                #             precisions.update(precision, len(data['img_id']))

                if i % 10 == 0:
                    print('Test: [{0}/{1}]\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                        i, len(val_loader), loss=losses, top1=top1))

            val_loss_arr.append(losses.avg)
            val_acc_arr.append(top1.avg)
            #         val_prec_arr.append(precisions.avg)
            print("Validation result: Loss: %.4f, Acc: %.4f" % (losses.avg, top1.avg))

if __name__ == "__main__":

    train_transform = transforms.Compose([
        transforms.Scale((224, 224)),
        #     transforms.RandomCrop(224),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    val_transform = transforms.Compose([
        transforms.Scale((224, 224)),
        #     transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    trainset = PDataset(csv_file="./dataset/2_scale_train_single_intersection.csv",
                        root_dir="./dataset/images/images", transform = train_transform)
    valset = PDataset(csv_file="./dataset/2_scale_test_single_intersection.csv",
                      root_dir="./dataset/images/images", transform = val_transform)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=False, num_workers=0)

    model = PNet(base_model = 'resnet50')
    loss_fn = FocalLoss(gamma = 2)
    train(train_loader, val_loader, model, loss_fn)