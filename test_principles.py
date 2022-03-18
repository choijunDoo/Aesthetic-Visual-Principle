import os

import numpy as np

from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import precision_score, average_precision_score, recall_score

import torchvision.transforms as transforms

from torch.utils.tensorboard import SummaryWriter

from load_dataset.dataset import PDataset

from model.model import *
from evalution_metrics import AverageMeter, accuracy, pred_acc
from transformers import ViTModel


import warnings
warnings.filterwarnings('ignore')

def test(test_loader, fold, method):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    if method == "Non":
        model = resnet2D56(non_local=True)
        model.load_state_dict(torch.load("./save/Union_{}_BCE_{}.pt".format(method, fold)))
    elif method == "GCN":
        model = GCN(nfeat=512, nhid=1024, nclass=8, dropout=0.5)
        model.load_state_dict(torch.load("./save/Union_{}_BCE_{}.pt".format(method, fold)))
    elif method == "ViT":
        pretrained_vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k', output_attentions=True)
        model = PreTrainedViT(pretrained_vit_model, 768, 8)
        model.load_state_dict(torch.load("./save/ViT_{}.pt".format(fold)))

    model = model.to(device)

    adj = torch.ones(7 * 7, 7 * 7)
    adj = adj.to(device)

    output_results = []
    labels_results = []
    count = 0

    model.eval()

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            images = data['image'].to(device)
            labels = data['information'].to(device)

            if method == "Non":
                outputs = model(images)
            elif method == "GCN":
                outputs = model(images, adj, len(data['image']))
            elif method == "ViT":
                outputs = model(images)[0]

            outputs = F.sigmoid(outputs)
            labels.float()
            outputs.float()

            output_results.append(outputs)
            labels_results.append(labels)

            count += 1

            if count % 10 == 0:
                print(count)

    output = torch.cat(output_results, 0)
    label = torch.cat(labels_results, 0)

    print(output.shape, label.shape)
    prec1 = pred_acc(output.data, label)
    micro_precision = precision_score(np.round(label.detach().cpu().numpy()), np.round(output.detach().cpu().numpy()),
                                      average='micro')
    macro_precision = precision_score(np.round(label.detach().cpu().numpy()), np.round(output.detach().cpu().numpy()),
                                      average='macro')
    micro_recall = recall_score(np.round(label.detach().cpu().numpy()), np.round(output.detach().cpu().numpy()),
                                average='micro')
    macro_recall = recall_score(np.round(label.detach().cpu().numpy()), np.round(output.detach().cpu().numpy()),
                                average='macro')

    print(prec1)
    print(micro_precision)
    print(macro_precision)
    print(micro_recall)
    print(macro_recall)

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

    P_dataset = PDataset(csv_file="./dataset/union.csv", root_dir="./dataset/images/images")

    k_folds = 5
    kfold = KFold(n_splits=k_folds, shuffle=False)

    for fold, (train_ids, test_ids) in enumerate(kfold.split(P_dataset)):

        print(f'FOLD {fold}')
        print('--------------------------------')

        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

        test_loader = torch.utils.data.DataLoader(
            PDataset(csv_file="./dataset/union.csv", root_dir="./dataset/images/images", transform=val_transform),
            batch_size=int(len(test_ids) / 100), sampler=test_subsampler)

        test(test_loader, fold, "ViT")
        #    acc_results[fold], micro_results[fold], macro_results[fold] = train(train_loader, val_loader, model, loss_fn, lr=1e-2, epochs=20)
        #
        #
        #    torch.save(model.state_dict(), "./save/Union_GCN_BCE_{}.pt".format(fold))
        #
        #
    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
    print('--------------------------------')