import os
import time
import random
import numpy as np
import logging
import pickle
import argparse
import collections
import sys
sys.path.append("..")
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('AGG')
# from util import config
from util.common_util import AverageMeter, intersectionAndUnion, check_makedirs
from util.voxelize import voxelize
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.decomposition import PCA
from tsnecuda import TSNE as TSNE_CUDA

random.seed(123)
np.random.seed(123)


def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"

    file_handler = logging.FileHandler('extract_data.log', mode='w')
    file_handler.setLevel(level=logging.INFO)
    file_handler.setFormatter(logging.Formatter(fmt))

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level=logging.INFO)
    stream_handler.setFormatter(logging.Formatter(fmt))

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def main():
    global args, logger
    logger = get_logger()
    logger.info(args)
    assert args.classes > 1
    logger.info("Classes: {}".format(args.classes))
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.test_gpu)


    extract_data()


def data_prepare(area):
    if args.data_name == 's3dis':
        data_list = sorted(os.listdir(args.data_root))
        data_list = [item[:-4] for item in data_list if 'Area_{}'.format(area) in item]
    else:
        raise Exception('dataset not supported yet'.format(args.data_name))
    print("Totally {} samples in val set.".format(len(data_list)))
    return data_list


def data_load(data_name):
    data_path = os.path.join(args.data_root, data_name + '.npy')
    data = np.load(data_path)  # xyzrgbl, N*7
    coord, feat, label = data[:, :3], data[:, 3:6], data[:, 6]

    return coord, feat, label


def extract_data():
    logger.info('>>>>>>>>>>>>>>>> Extract Data >>>>>>>>>>>>>>>>')

    # f = open('E:/test.txt', 'a')
    #
    # f.write('hello world!')
    #
    # f.close()
    args.batch_size_test = 10
    areas = [5]
    count_scene = np.zeros(args.classes)
    for area in areas:
        data_list = data_prepare(area)
        for idx, item in enumerate(data_list):
            logger.info(item)
            _, _, label = data_load(item)
            unique_label = np.unique(np.array(label))
            logger.info(unique_label)
            for label in unique_label:
                count_scene[int(label)] += 1
            print(count_scene)
    logger.info('Number of scenes: {}'.format(count_scene))
    logger.info('<<<<<<<<<<<<<<<<< End Extraction <<<<<<<<<<<<<<<<<')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Point Cloud Semantic Segmentation')
    parser.add_argument('--test_list', default='dataset/s3dis/list/val5.txt')
    parser.add_argument('--test_list_full', default='dataset/s3dis/list/val5_full.txt')
    parser.add_argument('--split', default='val')
    parser.add_argument('--test_gpu', default=[3])
    parser.add_argument('--test_workers', default=4)
    parser.add_argument('--batch_size_test', default=4)
    parser.add_argument('--model_path', default='/user/lijianan/point-transformer/exp/s3dis/pointtransformer_repro/model/model_best.pth')
    parser.add_argument('--names_path', default='/user/lijianan/point-transformer/data/s3dis/s3dis_names.txt')
    parser.add_argument('--data_name', default='s3dis')
    parser.add_argument('--data_root', default='/user/lijianan/point-transformer/data/stanford_indoor3d/')
    parser.add_argument('--classes', default=13)
    parser.add_argument('--fea_dim', default=6)
    parser.add_argument('--voxel_size', default=0.04)
    parser.add_argument('--voxel_max', default=80000)
    parser.add_argument('--loop', default=30)
    parser.add_argument('--arch', default='pointtransformer_seg_repro')
    parser.add_argument('--ignore_label', default=255)
    parser.add_argument('--save_folder', default='exp/s3dis/pointtransformer_repro/result/best')

    args = parser.parse_args()
    main()