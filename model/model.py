"""
file - model.py
Implements the aesthemic model and emd loss used in paper.
Copyright (C) Yunxiao Shi 2017 - 2021
NIMA is released under the MIT license. See LICENSE for the fill license text.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn.init as init
from torch.nn.init import xavier_uniform
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

import torchvision.models as models
from efficientnet_pytorch import EfficientNet


class PANet(nn.Module):
    def __init__(self, method, device, num_classes=10):
        super(PANet, self).__init__()

        self.backbone = models.vgg19(pretrained=True).features
        self.method = method
        self.device = device
        self.adj = torch.ones(49, 49)
        self.adj = self.adj.to(device)

        if self.method == "Non":
            self.principle = resnet2D56(non_local=True)
            self.principle.load_state_dict(torch.load("./save/Union_Non_BCE_0.pt"))

        elif self.method == "GCN":
            self.principle = GCN(512, 1024, 8, dropout=0.5)
            self.principle.load_state_dict(torch.load("./save/Union_GCN_BCE_0.pt"))

        self.set_parameter_requires_grad(self.principle, True)
        self.principle = self.principle.to(device)
        self.backbone = self.backbone.to(device)

        self.pool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc = nn.Linear(in_features=25235, out_features=num_classes)  # Non 25235, GCN 25480

    def forward(self, x, bs):

        if self.method == "Non":
            back_out = self.backbone(x)
            prin_out = self.principle.conv1(x)
            prin_out = self.principle.layer1(prin_out)
            prin_out = self.principle.layer2(prin_out)
            prin_out = self.principle.layer3(prin_out)
            prin_out = self.pool(prin_out)

        elif self.method == "GCN":
            prin_out = self.principle.backbone(x)
            prin_out = prin_out.view([bs, 512, -1])
            prin_out = prin_out.transpose(1,2)
            prin_out = F.relu(self.principle.gc1(prin_out, self.adj))
            prin_out = F.relu(self.principle.gc2(prin_out, self.adj))
            prin_out = F.relu(self.principle.gc2(prin_out, self.adj))
            prin_out = F.softmax(self.principle.gc3(prin_out, self.adj), dim=1)
            prin_out = prin_out.transpose(1,2)
            prin_out = prin_out.view([bs, -1, 7, 7])

        out = torch.cat([back_out, prin_out], dim=1)
        out = F.softmax(out, dim=1)
        out = out.view([bs, -1])
        out = self.fc(out)

        return out

class NIMA(nn.Module):

    """Neural IMage Assessment model by Google"""
    def __init__(self, base_model, num_classes=10):
        super(NIMA, self).__init__()
        self.features = base_model.features
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.75),
            nn.Linear(in_features=25088, out_features=num_classes),
            nn.Softmax())

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

class PreTrainedViT(nn.Module):
    def __init__(self, pretrained_vit_model, d_model, classes):
      super().__init__()
      self.pretrained_vit_model = pretrained_vit_model
      self.classifier = nn.Linear(d_model, classes)

    def forward(self, x):
      x = self.pretrained_vit_model(x)
      attentions = x.attentions
      output = self.classifier(x.last_hidden_state[:,0,:]) # cls tokken

      return output, attentions

class PNet(nn.Module):
    def __init__(self, num_classes=8):
        super(PNet, self).__init__()

        ## backbone
        # if base_model == "resnet34":
        #     self.model = models.resnet34(pretrained=False)
        #     self.model.fc = nn.Linear(in_features=512, out_features= num_classes)
        # elif base_model == "vgg16":
        #     self.model = models.vgg16(pretrained=True)
        #     self.model.classifier[6] = nn.Linear(in_features=4096, out_features=num_classes)

        # self.base_model = EfficientNet.from_pretrained("efficientnet-b0", num_classes=5)
        self.base_model = models.vgg19(pretrained=False)
        # self.set_parameter_requires_grad(self.base_model, True)
        # self.base_model.classifier[6] = nn.Linear(in_features=4096, out_features=num_classes)
        # self.base_model.fc = nn.Linear(in_features=4096, out_features= num_classes)
        # self.base_model.classifier[6] = nn.Linear(in_features=self.base_model.classifier[6].in_features, out_features=num_classes)
        self.features = self.base_model.features
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features = 25088, out_features=num_classes),
        )
        # self.sigmoid = nn.Sigmoid()
        self.weights_init()

    def forward(self, x):
        # out = self.model(x)
        # out = self.sigmoid(out)

        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)

        # out = self.base_model(x)

        return out

    def set_parameter_requires_grad(self, model, feature_extracting):
        if feature_extracting:
            for n, param in model.named_parameters():
                if 'classifier' not in n:
                    param.requires_grad = False

    def weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_uniform(m.weight.data)
            elif isinstance(m, nn.Linear):
                xavier_uniform(m.weight.data)


class NLBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, mode='embedded',
                 dimension=3, bn_layer=True):
        """Implementation of Non-Local Block with 4 different pairwise functions but doesn't include subsampling trick
        args:
            in_channels: original channel size (1024 in the paper)
            inter_channels: channel size inside the block if not specifed reduced to half (512 in the paper)
            mode: supports Gaussian, Embedded Gaussian, Dot Product, and Concatenation
            dimension: can be 1 (temporal), 2 (spatial), 3 (spatiotemporal)
            bn_layer: whether to add batch norm
        """
        super(NLBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        if mode not in ['gaussian', 'embedded', 'dot', 'concatenate']:
            raise ValueError('`mode` must be one of `gaussian`, `embedded`, `dot` or `concatenate`')

        self.mode = mode
        self.dimension = dimension

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        # the channel size is reduced to half inside the block
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        # assign appropriate convolutional, max pool, and batch norm layers for different dimensions
        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        # function g in the paper which goes through conv. with kernel size 1
        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        # add BatchNorm layer after the last conv layer
        if bn_layer:
            self.W_z = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1),
                bn(self.in_channels)
            )
            # from section 4.1 of the paper, initializing params of BN ensures that the initial state of non-local block is identity mapping
            nn.init.constant_(self.W_z[1].weight, 0)
            nn.init.constant_(self.W_z[1].bias, 0)
        else:
            self.W_z = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1)

            # from section 3.3 of the paper by initializing Wz to 0, this block can be inserted to any existing architecture
            nn.init.constant_(self.W_z.weight, 0)
            nn.init.constant_(self.W_z.bias, 0)

        # define theta and phi for all operations except gaussian
        if self.mode == "embedded" or self.mode == "dot" or self.mode == "concatenate":
            self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
            self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        if self.mode == "concatenate":
            self.W_f = nn.Sequential(
                nn.Conv2d(in_channels=self.inter_channels * 2, out_channels=1, kernel_size=1),
                nn.ReLU()
            )

    def forward(self, x):
        """
        args
            x: (N, C, T, H, W) for dimension=3; (N, C, H, W) for dimension 2; (N, C, T) for dimension 1
        """

        batch_size = x.size(0)

        # (N, C, THW)
        # this reshaping and permutation is from the spacetime_nonlocal function in the original Caffe2 implementation
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        if self.mode == "gaussian":
            theta_x = x.view(batch_size, self.in_channels, -1)
            phi_x = x.view(batch_size, self.in_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)

        elif self.mode == "embedded" or self.mode == "dot":
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
            phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)

        elif self.mode == "concatenate":
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1, 1)
            phi_x = self.phi(x).view(batch_size, self.inter_channels, 1, -1)

            h = theta_x.size(2)
            w = phi_x.size(3)
            theta_x = theta_x.repeat(1, 1, 1, w)
            phi_x = phi_x.repeat(1, 1, h, 1)

            concat = torch.cat([theta_x, phi_x], dim=1)
            f = self.W_f(concat)
            f = f.view(f.size(0), f.size(2), f.size(3))

        if self.mode == "gaussian" or self.mode == "embedded":
            f_div_C = F.softmax(f, dim=-1)
        elif self.mode == "dot" or self.mode == "concatenate":
            N = f.size(-1)  # number of position in x
            f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)

        # contiguous here just allocates contiguous chunk of memory
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])

        W_y = self.W_z(y)
        # residual connection
        z = W_y + x

        return z

def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        xavier_uniform(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant",
                                                  0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet2D(nn.Module):
    def __init__(self, block, num_blocks, num_classes=8, non_local=False):
        super(ResNet2D, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)

        # add non-local block after layer 2
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2, non_local=non_local)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride, non_local=False):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        last_idx = len(strides)
        if non_local:
            last_idx = len(strides) - 1

        for i in range(last_idx):
            layers.append(block(self.in_planes, planes, strides[i]))
            self.in_planes = planes * block.expansion

        if non_local:
            layers.append(NLBlockND(in_channels=planes, dimension=2))
            layers.append(block(self.in_planes, planes, strides[-1]))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet2D56(non_local=False, **kwargs):
    """Constructs a ResNet-56 model.
    """
    return ResNet2D(BasicBlock, [9, 9, 9], non_local=non_local, **kwargs)


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.backbone = models.vgg19(pretrained=False).features
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nclass)
        self.fc = nn.Linear(25088, 8)
        self.dropout = dropout

        # self.weights_init()

    def forward(self, x, adj, bs):
        x = self.backbone(x)
        x = x.view([bs, 512, -1])
        x = x.transpose(1, 2)

        x = F.softmax(self.gc1(x, adj), dim=1)
        # x = F.dropout(x, self.dropout, training=self.training)
        x = F.softmax(self.gc2(x, adj), dim=1)
        # x = F.relu(self.gc2(x, adj))
        # x = F.dropout(x, self.dropout, training=self.training)
        x = F.softmax(self.gc3(x, adj), dim=1)
        # x = F.log_softmax(x, dim=1)
        # x = x.view([64, -1])
        x = x.view([bs, -1])
        x = self.fc(x)

        return x

    def weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_uniform(m.weight.data)
            elif isinstance(m, nn.Linear):
                xavier_uniform(m.weight.data)