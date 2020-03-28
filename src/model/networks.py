# -*- coding: utf-8 -*-


import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class conv_module_cnn(nn.Module):
    def __init__(self):
        super(conv_module_cnn, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.batch_norm2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.batch_norm3 = nn.BatchNorm2d(32)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.batch_norm4 = nn.BatchNorm2d(32)
        self.relu4 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(self.batch_norm1(x))
        x = self.conv2(x)
        x = self.relu2(self.batch_norm2(x))
        x = self.conv3(x)
        x = self.relu3(self.batch_norm3(x))
        x = self.conv4(x)
        x = self.relu4(self.batch_norm4(x))
        return x.view(-1, 32 * 4 * 4)


class mlp_module_cnn(nn.Module):
    def __init__(self):
        super(mlp_module_cnn, self).__init__()
        self.fc1 = nn.Linear(32 * 4 * 4, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 99)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class CNN_MLP(nn.Module):
    def __init__(self):
        super(CNN_MLP, self).__init__()
        self.conv = conv_module_cnn()
        self.mlp = mlp_module_cnn()

    def forward(self, x):
        features = self.conv(x.view(-1, 3, 80, 80))
        score = self.mlp(features)
        return score


class conv_module_lstm(nn.Module):
    def __init__(self):
        super(conv_module_lstm, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2)
        self.batch_norm1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2)
        self.batch_norm2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=2)
        self.batch_norm3 = nn.BatchNorm2d(16)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(16, 16, kernel_size=3, stride=2)
        self.batch_norm4 = nn.BatchNorm2d(16)
        self.relu4 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(self.batch_norm1(x))
        x = self.conv2(x)
        x = self.relu2(self.batch_norm2(x))
        x = self.conv3(x)
        x = self.relu3(self.batch_norm3(x))
        x = self.conv4(x)
        x = self.relu4(self.batch_norm4(x))
        return x.view(-1, 3, 16 * 4 * 4)


class lstm_module(nn.Module):
    def __init__(self):
        super(lstm_module, self).__init__()
        self.lstm = nn.LSTM(input_size=16 * 4 * 4, hidden_size=256, num_layers=1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256, 99)

    def forward(self, x):
        x = x.permute(1, 0, 2)
        hidden, _ = self.lstm(x)
        score = self.fc(hidden[-1, :, :])
        return score


class CNN_LSTM(nn.Module):
    def __init__(self):
        super(CNN_LSTM, self).__init__()
        self.conv = conv_module_lstm()
        self.lstm = lstm_module()

    def forward(self, x):
        features = self.conv(x.view(-1, 1, 80, 80))
        score = self.lstm(features)
        return score


class identity(nn.Module):
    def __init__(self):
        super(identity, self).__init__()
    
    def forward(self, x):
        return x


class mlp_module_resnet(nn.Module):
    def __init__(self):
        super(mlp_module_resnet, self).__init__()
        self.fc1 = nn.Linear(512, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 99)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class Resnet18_MLP(nn.Module):
    def __init__(self):
        super(Resnet18_MLP, self).__init__()
        self.resnet18 = models.resnet18(pretrained=False)
        self.resnet18.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet18.fc = identity()
        self.mlp = mlp_module_resnet()

    def forward(self, x):
        features = self.resnet18(x.view(-1, 3, 224, 224))
        score = self.mlp(features)
        return score


class ConvInputModel(nn.Module):
    def __init__(self):
        super(ConvInputModel, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 24, 3, stride=2, padding=1)
        self.batchNorm1 = nn.BatchNorm2d(24)
        self.conv2 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.batchNorm2 = nn.BatchNorm2d(24)
        self.conv3 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.batchNorm3 = nn.BatchNorm2d(24)
        self.conv4 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.batchNorm4 = nn.BatchNorm2d(24)

        
    def forward(self, img):
        """convolution"""
        x = self.conv1(img)
        x = F.relu(x)
        x = self.batchNorm1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.batchNorm2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.batchNorm3(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.batchNorm4(x)
        return x


class FCOutputModel(nn.Module):
    def __init__(self):
        super(FCOutputModel, self).__init__()

        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 99)

    def forward(self, x):
        x = self.fc2(x)
        x = F.relu(x)
        x = F.dropout(x)
        x = self.fc3(x)
        return x


class RelationalNetwork(nn.Module):
    def __init__(self, batch_size=32, cuda=True):
        super(RelationalNetwork, self).__init__()
        self.cuda = cuda
        self.batch_size = batch_size

        self.conv = ConvInputModel()
        
        ##(number of filters per object + coordinate of object) * 2
        self.g_fc1 = nn.Linear((24 + 2) * 2, 256)
        self.g_fc2 = nn.Linear(256, 256)
        self.g_fc3 = nn.Linear(256, 256)
        self.g_fc4 = nn.Linear(256, 256)

        self.f_fc1 = nn.Linear(256, 256)

        self.coord_oi = torch.FloatTensor(self.batch_size, 2)
        self.coord_oj = torch.FloatTensor(self.batch_size, 2)
        if self.cuda:
            self.coord_oi = self.coord_oi.cuda()
            self.coord_oj = self.coord_oj.cuda()

        # prepare coord tensor
        def cvt_coord(i):
            return [(i / 5 - 2) / 2.0, (i % 5 - 2) / 2.0]
        
        self.coord_tensor = torch.FloatTensor(self.batch_size, 25, 2)
        if self.cuda:
            self.coord_tensor = self.coord_tensor.cuda()
        np_coord_tensor = np.zeros((self.batch_size, 25, 2))
        for i in range(25):
            np_coord_tensor[:, i, :] = np.array(cvt_coord(i))
        self.coord_tensor.data.copy_(torch.from_numpy(np_coord_tensor))
        self.fcout = FCOutputModel()

    def forward(self, img):
        x = self.conv(img) ## x = (64 x 24 x 5 x 5)

        """g"""
        mb = x.size()[0]
        n_channels = x.size()[1]
        d = x.size()[2]

        self.coord_oi = torch.FloatTensor(mb, 2)
        self.coord_oj = torch.FloatTensor(mb, 2)
        if self.cuda:
            self.coord_oi = self.coord_oi.cuda()
            self.coord_oj = self.coord_oj.cuda()

        # prepare coord tensor
        def cvt_coord(i):
            return [(i / 5 - 2) / 2.0, (i % 5 - 2) / 2.0]
        
        self.coord_tensor = torch.FloatTensor(mb, 25, 2)
        if self.cuda:
            self.coord_tensor = self.coord_tensor.cuda()
        np_coord_tensor = np.zeros((mb, 25, 2))
        for i in range(25):
            np_coord_tensor[:, i, :] = np.array(cvt_coord(i))
        self.coord_tensor.data.copy_(torch.from_numpy(np_coord_tensor))

        # x_flat = (64 x 25 x 24)
        x_flat = x.view(mb, n_channels, d * d).permute(0, 2, 1)
        
        # add coordinates
        x_flat = torch.cat([x_flat, self.coord_tensor], 2)
        
        # cast all pairs against each other
        x_i = torch.unsqueeze(x_flat, 1) # (64x1x25x26+11)
        x_i = x_i.repeat(1, 25, 1, 1) # (64x25x25x26+11)
        x_j = torch.unsqueeze(x_flat, 2) # (64x25x1x26+11)
        x_j = x_j.repeat(1, 1, 25, 1) # (64x25x25x26+11)
        
        # concatenate all together
        x_full = torch.cat([x_i, x_j], 3) # (64x25x25x2*26+11)

        # reshape for passing through network
        x_ = x_full.view(mb * d * d * d * d, 52)
        x_ = self.g_fc1(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc2(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc3(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc4(x_)
        x_ = F.relu(x_)
        
        # reshape again and sum
        x_g = x_.view(mb, d * d * d * d, 256)
        x_g = x_g.sum(1).squeeze()
        
        """f"""
        x_f = self.f_fc1(x_g)
        x_f = F.relu(x_f)
        
        return self.fcout(x_f)