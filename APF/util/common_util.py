import os

import numpy as np
import torch
import pdb


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def intersectionAndUnion(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.ndim in [1, 2, 3])
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    # output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K+1))
    area_output, _ = np.histogram(output, bins=np.arange(K+1))
    area_target, _ = np.histogram(target, bins=np.arange(K+1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def intersectionAndUnionGPU(output, target, K, ignore_index=255, unknown_clss=[0, 1]):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.dim() in [1, 2, 3])
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    # print('-------------------------target----------------------')
    # instances, counts = np.unique(target.cpu().numpy(), False, False, True)
    # print(instances)
    # print(counts)
    # output[target == ignore_index] = ignore_index
    intersection = output[output == target]

    if ignore_index == 0:
        # print('闭集')
        area_intersection = torch.histc(intersection, bins=K-len(unknown_clss), min=len(unknown_clss), max=K - 1)
        area_output = torch.histc(output, bins=K-len(unknown_clss), min=len(unknown_clss), max=K - 1)
        area_target = torch.histc(target, bins=K-len(unknown_clss), min=len(unknown_clss), max=K - 1)
        # print('intersection', area_intersection)
        # print('output', area_output)
        # print('target', area_target)
    else:
        # print('开集')
        area_intersection = torch.histc(intersection, bins=K, min=0, max=K - 1)
        area_output = torch.histc(output, bins=K, min=0, max=K - 1)
        area_target = torch.histc(target, bins=K, min=0, max=K - 1)
        # print('intersection', area_intersection)
        # print('output', area_output)
        # print('target', area_target)
        # pdb.set_trace()

    area_union = area_output + area_target - area_intersection
    # if ignore_index != 0:
    #     print('output', area_output, sum(area_output))
    #     print('target', area_target, sum(area_target))
    #     print('union', area_union)
    #     print('intersection', area_intersection)
    #     iou_class = area_intersection / (area_union + 1e-10)
    #     print(iou_class)
    #     import pdb
    #     pdb.set_trace()
    return area_intersection, area_union, area_target


def check_makedirs(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def find_free_port():
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port
