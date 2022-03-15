import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import winnt
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import precision_score, average_precision_score

import torch
import torch.autograd as autograd
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torchvision.models as models

from torch.utils.tensorboard import SummaryWriter

from load_dataset.dataset import PDataset
from losses import focal_binary_cross_entropy, FocalLoss, AsymmetricLoss, CBLoss

from model.model import *
from evalution_metrics import AverageMeter, accuracy, pred_acc
from model.CBAM import ResidualNet
from torchmetrics import HammingDistance, HingeLoss

import warnings
warnings.filterwarnings('ignore')

def train(train_loader, val_loader, model, loss_fn, lr = 1e-4, epochs = 50, k_folds = 5, decay = 'store_true', lr_decay_rate = 0.95, lr_decay_freq = 10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # writer = SummaryWriter()

    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), momentum=0.9, lr=lr, weight_decay=0.1)

    param_num = 0
    for param in model.parameters():
        if param.requires_grad:
            param_num += param.numel()
    print('Trainable params: %.2f million' % (param_num / 1e6))

    train_loss_arr = []
    train_acc_arr = []
    train_micro_prec_arr = [] # multi_label
    train_macro_prec_arr = []  # multi_label

    val_loss_arr = []
    val_acc_arr = []
    val_micro_prec_arr = [] #  multi-label
    val_macro_prec_arr = []  # multi-label

    adj = torch.ones(7 * 7, 7 * 7)
    adj = adj.to(device)

    for epoch in range(0, epochs):

        model.train()

        losses = AverageMeter()
        top1 = AverageMeter()
        micro_precisions = AverageMeter()
        macro_precisions = AverageMeter()

        for i, data in enumerate(train_loader):
            images = data['image'].to(device)
            labels = data['information'].to(device).float() ## (batch 128, 8)

            outputs = model(images, adj, len(data['img_id']))  # (batch, 128, 8)
            outputs = F.sigmoid(outputs)
            # _, labels = torch.max(labels, 1)
            # outputs = outputs.view(-1, 8, 1)
            optimizer.zero_grad()
            outputs.float()
            loss = loss_fn(outputs, labels)  ##
            loss.float()

            prec1 = pred_acc(outputs.data, labels)  ##
            # prec1 = prec1[0]

            micro_precision = precision_score(np.round(labels.detach().cpu().numpy()), np.round(outputs.detach().cpu().numpy()),
                                              average='micro')
            macro_precision = precision_score(np.round(labels.detach().cpu().numpy()), np.round(outputs.detach().cpu().numpy()),
                                              average='macro')
            losses.update(loss.item(), len(data['img_id']))
            top1.update(prec1, len(data['img_id']))
            micro_precisions.update(micro_precision, len(data['img_id']))
            macro_precisions.update(macro_precision, len(data['img_id']))

            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(epoch, i, len(train_loader), loss=losses))
        #         writer.add_scalar('batch train loss', loss.data, i + epoch * (len(trainset) // 128 + 1))

        train_loss_arr.append(losses.avg)
        train_acc_arr.append(top1.avg)
        train_micro_prec_arr.append(micro_precisions.avg)
        train_macro_prec_arr.append(macro_precisions.avg)

        print("train result: Loss: %.4f, Acc: %.4f, Micro-Prec: %.4f, Macro-Prec: %.4f"
              % (losses.avg, top1.avg, micro_precisions.avg, macro_precisions.avg))
        # print("train result: Loss: %.4f, Acc: %.4f" % (losses.avg, top1.avg))

        # exponetial learning rate decay
        if decay:
            if (epoch + 1) % 10 == 0:
                lr = lr * lr_decay_rate ** ((epoch + 1) / lr_decay_freq)
                # dense_lr = dense_lr * lr_decay_rate ** ((epoch + 1) / lr_decay_freq)
                # optimizer = optim.SGD([
                #     {'params': model.features.parameters(), 'lr': conv_base_lr},
                #     {'params': model.classifier.parameters(), 'lr': dense_lr}],
                #     momentum=0.9
                # )
                optimizer = optim.SGD(model.parameters(), momentum=0.9, lr=lr, weight_decay=0.1)

        model.eval()
        with torch.no_grad():

            losses = AverageMeter()
            top1 = AverageMeter()
            micro_precisions = AverageMeter()
            macro_precisions = AverageMeter()

            for i, data in enumerate(val_loader):
                images = data['image'].to(device)
                labels = data['information'].to(device) ## (batch 128, 8)
                outputs = model(images, adj, len(data['img_id']))  # (batch, 128, 8)
                outputs = F.sigmoid(outputs)
                # _, labels = torch.max(labels, 1)
                labels = labels.float()

                outputs.float()
                loss = loss_fn(outputs, labels)
                loss.float()

                prec1 = pred_acc(outputs.data, labels)
                # prec1 = prec1[0]
                micro_precision = precision_score(np.round(labels.detach().cpu().numpy()), np.round(outputs.detach().cpu().numpy()),
                                                  average='micro')
                macro_precision = precision_score(np.round(labels.detach().cpu().numpy()), np.round(outputs.detach().cpu().numpy()),
                                                  average='macro')

                losses.update(loss.item(), len(data['img_id']))
                top1.update(prec1, len(data['img_id']))
                micro_precisions.update(micro_precision, len(data['img_id']))
                macro_precisions.update(macro_precision, len(data['img_id']))

                if i % 10 == 0:
                    print('Test: [{0}/{1}]\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Micro-Prec {micro_prec.val:.3f} ({micro_prec.avg:.3f})\t'
                          'Macro-Prec {macro_prec.val:.3f} ({macro_prec.avg:.3f})\t'.format(
                        i, len(val_loader), loss=losses, top1=top1, micro_prec = micro_precisions, macro_prec = macro_precisions))

                    # print('Test: [{0}/{1}]\t'
                    #       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    #       'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                    #     i, len(val_loader), loss=losses, top1=top1))

            val_loss_arr.append(losses.avg)
            val_acc_arr.append(top1.avg)
            val_micro_prec_arr.append(micro_precisions.avg)
            val_macro_prec_arr.append(micro_precisions.avg)
            print("Validation result: Loss: %.4f, Acc: %.4f, Micro-Prec: %.4f, Macro-Prec: %.4f" % (losses.avg, top1.avg, micro_precisions.avg, macro_precisions.avg))
            # print("Validation result: Loss: %.4f, Acc: %.4f" % (losses.avg, top1.avg))

    return top1.avg, micro_precisions.avg, macro_precisions.avg


if __name__ == "__main__":


    train_transform = transforms.Compose([
        transforms.Scale((224, 224)),
        # transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    val_transform = transforms.Compose([
        transforms.Scale((224, 224)),
        #     transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    P_dataset = PDataset(csv_file="./dataset/union.csv",
                         root_dir="./dataset/images/images")

    # trainset = PDataset(csv_file="./dataset/train_crop.csv",
    #                     root_dir="./dataset/crop_dataset", transform = train_transform)
    # valset = PDataset(csv_file="./dataset/test_crop.csv",
    #                   root_dir="./dataset/crop_dataset", transform = val_transform)

    # train_loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=0)
    # val_loader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=False, num_workers=0)

    # # For fold results
    acc_results = {}
    micro_results = {}
    macro_results = {}

    k_folds = 5
    kfold = KFold(n_splits=k_folds, shuffle=False)

    # loss_fn = CBLoss(samples_per_cls = [842, 997, 298, 223, 257, 195, 321, 252], no_of_classes = 8, loss_type = "focal", beta = 0.9999, gamma = 2.0)
    loss_fn = nn.BCELoss()
    # model = resnet2D56(non_local=True)

    # train_loader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True)
    # val_loader = torch.utils.data.DataLoader(valset, batch_size=4, shuffle=False)
    #
    # train(train_loader, val_loader, model, loss_fn, lr=1e-3, epochs=20)
    # torch.save(model.state_dict(), "./save/Crop_L2_BCE.pt")


    for fold, (train_ids, test_ids) in enumerate(kfold.split(P_dataset)):
        model = GCN(512, 1024, 8, dropout=0.5)
        print(f'FOLD {fold}')
        print('--------------------------------')

        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

        # Define data loaders for training and testing data in this fold
        train_loader = torch.utils.data.DataLoader(
            PDataset(csv_file="./dataset/union.csv",
                     root_dir="./dataset/images/images", transform = train_transform),
            batch_size = 64, sampler = train_subsampler)

        val_loader = torch.utils.data.DataLoader(
            PDataset(csv_file="./dataset/union.csv",
                     root_dir="./dataset/images/images", transform = val_transform),
            batch_size = 64, sampler = test_subsampler)

        acc_results[fold], micro_results[fold], macro_results[fold] = train(train_loader, val_loader, model, loss_fn, lr=1e-2, epochs = 50)

        torch.save(model.state_dict(), "./save/Non_Union_BCE.pt".format(fold))

    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
    print('--------------------------------')
    sum = 0.0

    for key, value in acc_results.items():
        print(f'Fold {key}: {value} %')
        sum += value
    print(f'Average: {sum / len(acc_results.items())} %')

    sum = 0.0

    for key, value in micro_results.items():
        print(f'Fold {key}: {value} %')
        sum += value
    print(f'Average: {sum / len(micro_results.items())} %')

    sum = 0.0

    for key, value in micro_results.items():
        print(f'Fold {key}: {value} %')
        sum += value
    print(f'Average: {sum / len(micro_results.items())} %')
