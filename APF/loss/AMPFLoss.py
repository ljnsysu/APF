import torch
import torch.nn as nn
import torch.nn.functional as f
from loss.Dist import Dist
import pdb
from torch.autograd import Variable


class AMPFLoss(nn.CrossEntropyLoss):
    def __init__(self, num_classes=11, feat_dim=32, lam=0.1):
        super(AMPFLoss, self).__init__()
        # self.use_gpu = options['use_gpu']
        self.weight_pl = float(lam)
        self.Dist = Dist(num_classes=num_classes, feat_dim=feat_dim)
        self.points = nn.Parameter(self.Dist.centers)
        self.open_proto = nn.Parameter(0.1*torch.randn(1, feat_dim)).cuda()
        self.radius = nn.Parameter(torch.Tensor(1))
        self.radius.data.fill_(0)
        self.margin_loss = nn.MarginRankingLoss(margin=0)

    def forward(self, x, labels=None):
        # dist = self.Dist(x, center=self.points)

        # --------------------------------------------------------------------------------------------------------------
        # x = x[labels != 0]
        # y = y[labels != 0]
        centers = torch.cat((self.points, self.open_proto), dim=0)
        dist_dot_p = self.Dist(x, center=centers, metric='dot')
        dist_l2_p = self.Dist(x, center=centers)
        logits = -1*(dist_l2_p - dist_dot_p)           # [batch_size, number_class]
        # logits = -dist_l2_p

        proto_loss = torch.abs(torch.ones_like(centers).cuda() - centers).mean()
        # logits = -dist_l2_p

        if labels is None:
            return logits, 0
        loss_main = f.cross_entropy(logits[labels != 0], labels[labels != 0] - 2)

        # -----------------------写法1---------------------
        # loss_var = (dist_l2_p[labels != 0, labels[labels != 0] - 2]).mean(0)

        # -----------------------写法2---------------------
        loss_var = Variable(torch.Tensor([0])).cuda()
        for instance in torch.unique(labels):
            if instance == 0:
                continue
            vectors = dist_l2_p[labels == instance]
            loss_var += torch.sum((vectors[:, int(instance) - 2])) / labels[labels != 0].shape[0]

        # mask = torch.ones(dist_l2_p.shape).cuda()
        # # 开集的点不参与计算
        # mask[labels == 0, :] = 0
        # # 闭集的点对应的label位置不参与计算
        # mask[labels != 0, labels[labels != 0] - 2] = 0
        # dist_masked = dist_l2_p * mask
        # close_dist, _ = torch.sort(dist_masked, dim=1)
        # close_dist = close_dist[labels != 0, 1]
        # correct_dist = dist_l2_p[labels != 0, labels[labels != 0] - 2]
        # # loss_push = close_dist.mean(0)
        # loss_push = nn.functional.relu(((correct_dist - close_dist) / (close_dist + correct_dist)) + 1).mean(0)
        # loss_push = torch.clamp(1 - loss_push, min=0)

        # loss_push = ((dist_l2_p * mask).sum(1)).sum() / (labels[labels != 0].shape[0])
        # loss_push = torch.clamp(5 + loss_push, min=0)

        # center_batch = self.points[labels[labels != 0] - 2, :]
        # _dis_known = (x[labels != 0] - center_batch).pow(2).mean(1)
        # target = torch.ones(_dis_known.size()).cuda()
        # loss_r = self.margin_loss(self.radius, _dis_known, target)
        # loss(x1,x2,y)=max(0,−y∗(x1−x2)+margin)    max(0, d_e(C(x), P^k) - R + 1)

        loss = loss_main + (loss_var + proto_loss) * 0.1
        # loss = loss_main + loss_var * 0.1
        # loss = loss_main
        # loss = 0
        return logits, loss, self.Dist.centers, dist_l2_p

    def fake_loss(self, x):
        logits = self.Dist(x, center=self.points)
        prob = f.softmax(logits, dim=1)
        loss = (prob * torch.log(prob + 1e-10)).sum(1).mean().exp()
        # loss = logits.sum(1).mean()
        # loss = loss + logits.mean() * 0.1

        return loss

    def fake_min_loss(self, x):
        logits = self.Dist(x, center=self.points)
        loss = logits.sum(1).mean()
        return loss

    def dce_loss(self, dist, labels, T):
        logits = -dist / T
        mean_loss = f.cross_entropy(logits[labels != 0], labels[labels != 0] - 2)
        return mean_loss

    def mcl_loss(self, dist, labels, margin):
        mask = torch.ones(dist.shape).cuda()
        # 开集的点不参与计算
        mask[labels == 0, :] = 0
        # 闭集的点对应的label位置不参与计算
        mask[labels != 0, labels[labels != 0] - 2] = 0
        dist_masked = dist * mask
        close_dist, _ = torch.sort(dist_masked, dim=1)
        close_dist = close_dist[labels != 0, 1]
        correct_dist = dist[labels != 0, labels[labels != 0] - 2]
        # loss_push = close_dist.mean(0)
        loss_push = nn.functional.relu(((correct_dist - close_dist) / (close_dist + correct_dist)) + 1).mean(0)
        loss_push = torch.clamp(margin - loss_push, min=0)
        return loss_push

    def prototype_loss(self, features, labels, centers):
        batch_num = labels[labels != 0].shape[0]
        index = labels[labels != 0] - 2
        index = index.unsqueeze(1)
        index = index.expand(index.shape[0], centers.shape[1])
        batch_centers = torch.gather(centers, 0, index)
        dis = features[labels != 0] - batch_centers
        proto_loss = (torch.sum(dis ** 2) / 2.0) / batch_num
        return proto_loss

    def forward_cpn(self, features, labels):
        dist = self.Dist(features, center=self.points)
        loss_dce = self.dce_loss(dist, labels, 1.0)
        loss_mcl = self.mcl_loss(dist, labels, 1.0)
        loss_proto = self.prototype_loss(features, labels, self.points)
        loss = loss_dce + loss_mcl + 0.001 * loss_proto
        return -dist, loss, self.Dist.centers, dist

    # def open_proto_loss(self, gen_features):



# loss = AMPFLoss()
# for p in loss.parameters():
#     print(p)
