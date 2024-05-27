# -*- coding:utf-8 -*-
# author: Xinge
# @file: metric_util.py 

import numpy as np


def fast_hist(pred, label, n):
    k = ((label >= 0) & (label < n)).cpu().numpy()
    bin_count = np.bincount(n * label[k].cpu().numpy() + pred[k], minlength=n ** 2)
    return bin_count[:n ** 2].reshape(n, n)


def per_class_iu(hist):
    np.seterr(divide='ignore', invalid='ignore')
    iou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    np.seterr(divide='warn', invalid='warn')
    iou[np.isnan(iou)] = 0
    return iou


def fast_hist_crop(output, target, unique_label):
    hist = fast_hist(output.flatten(), target.flatten(), np.max(unique_label) + 1)
    # print('hist', hist)
    # import pdb
    # pdb.set_trace()
    # hist = hist[unique_label+1, :]
    # hist = hist[:, unique_label+1]
    return hist
