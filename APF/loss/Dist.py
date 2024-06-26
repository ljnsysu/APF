import torch
import torch.nn as nn
import pdb


class Dist(nn.Module):      # to compute the distance, such as: d_e(x,y) or d_d(x,y)
    def __init__(self, num_classes=11, num_centers=1, feat_dim=2, init='random'):
        super(Dist, self).__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.num_centers = num_centers  # 每个类别的center数目

        if init == 'random':
            self.centers = nn.Parameter(0.1*torch.randn(num_classes * num_centers, self.feat_dim)).cuda()
        elif init == 'fix':
            self.centers = torch.zeros(num_classes, num_classes).cuda()
            for i in range(num_classes):
                self.centers[i][i] = 1
            print(self.centers)
        else:
            self.centers = nn.Parameter(torch.Tensor(num_classes * num_centers, self.feat_dim)).cuda()
            self.centers.data.fill_(0)

    def forward(self, features, center=None, metric='l2'):
        if metric == 'l2':
            f_2 = torch.sum(torch.pow(features, 2), dim=1, keepdim=True)
            if center is None:
                c_2 = torch.sum(torch.pow(self.centers, 2), dim=1, keepdim=True)
                dist = f_2 - 2*torch.matmul(features, torch.transpose(self.centers, 1, 0)) + torch.transpose(c_2, 1, 0)
            else:               # default compute: d_e(features, P^k)
                c_2 = torch.sum(torch.pow(center, 2), dim=1, keepdim=True)
                dist = f_2 - 2*torch.matmul(features, torch.transpose(center, 1, 0)) + torch.transpose(c_2, 1, 0)
            # dist = dist / float(features.shape[1])
        else:
            if center is None:
                center = self.centers 
            else:
                center = center
            dist = features.matmul(center.t())
        if dist.shape[1] == self.num_classes:
            dist = torch.reshape(dist, [-1, self.num_classes, self.num_centers])   # [batch_size, number_class, num_centers]
        else:
            dist = torch.reshape(dist, [-1, self.num_classes+1, self.num_centers])  # [batch_size, number_class, num_centers]
        dist = torch.mean(dist, dim=2)                                          # [batch_size, number_class]
        return dist
