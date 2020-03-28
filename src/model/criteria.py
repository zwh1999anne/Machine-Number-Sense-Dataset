# -*- coding: utf-8 -*-


import torch
import torch.nn.functional as F


def calculate_acc(output, target):
    pred = output.data.max(1)[1]
    correct = pred.eq(target.data).cpu().sum().numpy()
    return correct * 100.0 / target.size()[0]


def calculate_correct(output, target):
    pred = output.data.max(1)[1]
    correct = pred.eq(target.data).cpu().sum().numpy()
    return correct


def cross_entropy(output, target):
    return F.cross_entropy(output, target)
