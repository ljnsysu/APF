import torch
import torch.nn as nn
import pdb
from lib.pointops.functions import pointops
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from model.ABN import MultiBatchNorm


class PointTransformerLayer(nn.Module):
    def __init__(self, in_planes, out_planes, share_planes=8, nsample=16):
        # in_planes: fea_dim
        # out_planes: num_classes
        super().__init__()
        self.mid_planes = mid_planes = out_planes // 1
        self.out_planes = out_planes
        self.share_planes = share_planes
        self.nsample = nsample
        self.linear_q = nn.Linear(in_planes, mid_planes)
        self.linear_k = nn.Linear(in_planes, mid_planes)
        self.linear_v = nn.Linear(in_planes, out_planes)
        self.linear_p = nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3), nn.ReLU(inplace=True),
                                      nn.Linear(3, out_planes))
        self.linear_w = nn.Sequential(nn.BatchNorm1d(mid_planes), nn.ReLU(inplace=True),
                                      nn.Linear(mid_planes, mid_planes // share_planes),
                                      nn.BatchNorm1d(mid_planes // share_planes), nn.ReLU(inplace=True),
                                      nn.Linear(out_planes // share_planes, out_planes // share_planes))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, pxo) -> torch.Tensor:
        p, x, o = pxo  # (n, 3), (n, c), (b)
        # print(p.shape, x.shape, o.shape)  # torch.Size([110327, 3]) torch.Size([110327, 32]) torch.Size([4])
        x_q, x_k, x_v = self.linear_q(x), self.linear_k(x), self.linear_v(x)  # (n, c)
        x_k = pointops.queryandgroup(self.nsample, p, p, x_k, None, o, o, use_xyz=True)  # (n, nsample, 3+c)
        x_v = pointops.queryandgroup(self.nsample, p, p, x_v, None, o, o, use_xyz=False)  # (n, nsample, c)
        p_r, x_k = x_k[:, :, 0:3], x_k[:, :, 3:]
        for i, layer in enumerate(self.linear_p):
            p_r = layer(p_r.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i == 1 else layer(
                p_r)  # (n, nsample, c)
        w = x_k - x_q.unsqueeze(1) + p_r.view(p_r.shape[0], p_r.shape[1], self.out_planes // self.mid_planes,
                                              self.mid_planes).sum(2)  # (n, nsample, c)
        for i, layer in enumerate(self.linear_w):
            w = layer(w.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i % 3 == 0 else layer(w)
        # print(w.shape)  # torch.Size([110327, 8, 4])
        # pdb.set_trace()
        w = self.softmax(w)  # (n, nsample, c)
        n, nsample, c = x_v.shape;
        s = self.share_planes
        x = ((x_v + p_r).view(n, nsample, s, c // s) * w.unsqueeze(2)).sum(1).view(n, c)
        return x


class TransitionDown(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, nsample=16):
        super().__init__()
        self.stride, self.nsample = stride, nsample
        if stride != 1:
            self.linear = nn.Linear(3 + in_planes, out_planes, bias=False)
            self.pool = nn.MaxPool1d(nsample)
        else:
            self.linear = nn.Linear(in_planes, out_planes, bias=False)
        self.bn = nn.BatchNorm1d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pxo):
        p, x, o = pxo  # (n, 3), (n, c), (b)
        if self.stride != 1:
            n_o, count = [o[0].item() // self.stride], o[0].item() // self.stride
            for i in range(1, o.shape[0]):
                count += (o[i].item() - o[i - 1].item()) // self.stride
                n_o.append(count)
            n_o = torch.cuda.IntTensor(n_o)
            idx = pointops.furthestsampling(p, o, n_o)  # (m)
            n_p = p[idx.long(), :]  # (m, 3)
            x = pointops.queryandgroup(self.nsample, p, n_p, x, None, o, n_o, use_xyz=True)  # (m, 3+c, nsample)
            x = self.relu(self.bn(self.linear(x).transpose(1, 2).contiguous()))  # (m, c, nsample)
            x = self.pool(x).squeeze(-1)  # (m, c)
            p, o = n_p, n_o
        else:
            x = self.relu(self.bn(self.linear(x)))  # (n, c)
        return [p, x, o]


class TransitionUp(nn.Module):
    def __init__(self, in_planes, out_planes=None):
        super().__init__()
        if out_planes is None:
            self.linear1 = nn.Sequential(nn.Linear(2 * in_planes, in_planes), nn.BatchNorm1d(in_planes),
                                         nn.ReLU(inplace=True))
            self.linear2 = nn.Sequential(nn.Linear(in_planes, in_planes), nn.ReLU(inplace=True))
        else:
            self.linear1 = nn.Sequential(nn.Linear(out_planes, out_planes), nn.BatchNorm1d(out_planes),
                                         nn.ReLU(inplace=True))
            self.linear2 = nn.Sequential(nn.Linear(in_planes, out_planes), nn.BatchNorm1d(out_planes),
                                         nn.ReLU(inplace=True))

    def forward(self, pxo1, pxo2=None):
        if pxo2 is None:
            _, x, o = pxo1  # (n, 3), (n, c), (b)
            x_tmp = []
            for i in range(o.shape[0]):
                if i == 0:
                    s_i, e_i, cnt = 0, o[0], o[0]
                else:
                    s_i, e_i, cnt = o[i - 1], o[i], o[i] - o[i - 1]
                x_b = x[s_i:e_i, :]
                x_b = torch.cat((x_b, self.linear2(x_b.sum(0, True) / cnt).repeat(cnt, 1)), 1)
                x_tmp.append(x_b)
            x = torch.cat(x_tmp, 0)
            x = self.linear1(x)
        else:
            p1, x1, o1 = pxo1;
            p2, x2, o2 = pxo2
            x = self.linear1(x1) + pointops.interpolation(p2, p1, self.linear2(x2), o2, o1)
        return x


class PointTransformerBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, share_planes=8, nsample=16):
        super(PointTransformerBlock, self).__init__()
        self.linear1 = nn.Linear(in_planes, planes, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.transformer2 = PointTransformerLayer(planes, planes, share_planes, nsample)
        self.bn2 = nn.BatchNorm1d(planes)
        self.linear3 = nn.Linear(planes, planes * self.expansion, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pxo):
        p, x, o = pxo  # (n, 3), (n, c), (b)
        identity = x
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.relu(self.bn2(self.transformer2([p, x, o])))
        x = self.bn3(self.linear3(x))
        x += identity
        x = self.relu(x)
        return [p, x, o]


def weights_init_ABN(m):
    classname = m.__class__.__name__
    # TODO: what about fully-connected layers?
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.05)
    elif classname.find('MultiBatchNorm') != -1:
        m.bns[0].weight.data.normal_(1.0, 0.02)
        m.bns[0].bias.data.fill_(0)
        m.bns[1].weight.data.normal_(1.0, 0.02)
        m.bns[1].bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class PointTransformerSeg(nn.Module):
    def __init__(self, block, blocks, freeze_encoder=False, freeze_decoder=False, c=6, k=13):
        # block: PointTransformer
        # blocks: number of PointTransformer block [2, 3, 4, 6, 3]
        super().__init__()
        # c: original feature dim (RGB)
        self.c = c
        self.k = k
        self.unknown_clss = [0, 1]
        self.freeze_encoder = freeze_encoder
        self.freeze_decoder = freeze_decoder
        self.in_dim = 32
        self.fea_dim = 32
        self.memo_dim = 11

        # in_planes:input feature dim
        # planes: output feature dim
        self.in_planes, planes = c, [32, 64, 128, 256, 512]
        fpn_planes, fpnhead_planes, share_planes = 128, 64, 8
        stride, nsample = [1, 4, 4, 4, 4], [8, 16, 16, 16, 16]
        self.enc1 = self._make_enc(block, planes[0], blocks[0], share_planes, stride=stride[0],
                                   nsample=nsample[0])  # N/1
        self.enc2 = self._make_enc(block, planes[1], blocks[1], share_planes, stride=stride[1],
                                   nsample=nsample[1])  # N/4
        self.enc3 = self._make_enc(block, planes[2], blocks[2], share_planes, stride=stride[2],
                                   nsample=nsample[2])  # N/16
        self.enc4 = self._make_enc(block, planes[3], blocks[3], share_planes, stride=stride[3],
                                   nsample=nsample[3])  # N/64
        self.enc5 = self._make_enc(block, planes[4], blocks[4], share_planes, stride=stride[4],
                                   nsample=nsample[4])  # N/256
        # 冻结Encoder的参数
        if self.freeze_encoder:
            for param in self.parameters():
                param.requires_grad = False

        self.dec5 = self._make_dec(block, planes[4], 2, share_planes, nsample=nsample[4], is_head=True)  # transform p5
        self.dec4 = self._make_dec(block, planes[3], 2, share_planes, nsample=nsample[3])  # fusion p5 and p4
        self.dec3 = self._make_dec(block, planes[2], 2, share_planes, nsample=nsample[2])  # fusion p4 and p3
        self.dec2 = self._make_dec(block, planes[1], 2, share_planes, nsample=nsample[1])  # fusion p3 and p2
        self.dec1 = self._make_dec(block, planes[0], 2, share_planes, nsample=nsample[0])  # fusion p2 and p1

        # 冻结Decoder的参数
        # 当需冻结参数时，在定义优化器时也需告诉优化器哪些需要更新，哪些不需要（使用filter）
        # optimizer.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr)
        if self.freeze_decoder:
            for param in self.parameters():
                param.requires_grad = False

        self.logits1 = nn.Sequential(nn.Linear(planes[0], planes[0]), nn.BatchNorm1d(planes[0]), nn.ReLU(inplace=True),
                                     nn.Linear(planes[0], 13), nn.LeakyReLU(inplace=True))
        self.logits2 = nn.Sequential(nn.Linear(planes[0], planes[0]), nn.BatchNorm1d(planes[0]), nn.ReLU(inplace=True),
                                     nn.Linear(planes[0], 2))
        self.ln1 = nn.Linear(planes[0], planes[0])
        self.bn1 = nn.BatchNorm1d(planes[0])
        self.bn2 = nn.BatchNorm1d(planes[0])
        self.mbn1 = MultiBatchNorm(32, 2)
        self.ac = nn.ReLU(inplace=True)
        self.ln2 = nn.Linear(planes[0], 32)
        self.mbn2 = MultiBatchNorm(32, 2)
        # self.logits2 = nn.Sequential(nn.Linear(planes[0], planes[1]), MultiBatchNorm(64, 2), nn.ReLU(inplace=True),
        #                              nn.Linear(planes[1], 32))
        # self.logits2 = nn.Sequential(nn.Linear(32, 64),
        #     nn.ReLU(inplace=True),
        #
        #     nn.Linear(64, 32),
        #     nn.ReLU(inplace=True),
        #
        #     nn.Linear(32, 32))

        init_proto = torch.zeros(self.memo_dim, self.fea_dim)
        nn.init.kaiming_uniform_(init_proto)
        self.memory = nn.Parameter(init_proto)
        self.cosine_similarity = nn.CosineSimilarity(dim=2, )
        self.threshold = 1 / self.memory.size(0)
        self.epsilon = 1e-15
        self.addressing = 'hard'
        self.apply(weights_init_ABN)

    def _make_enc(self, block, planes, blocks, share_planes=8, stride=1, nsample=16):
        # in_planes:input feature dim
        # planes: output feature dim
        layers = []
        layers.append(TransitionDown(self.in_planes, planes * block.expansion, stride, nsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, self.in_planes, share_planes, nsample=nsample))
        return nn.Sequential(*layers)

    def _make_dec(self, block, planes, blocks, share_planes=8, nsample=16, is_head=False):
        layers = []
        layers.append(TransitionUp(self.in_planes, None if is_head else planes * block.expansion))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, self.in_planes, share_planes, nsample=nsample))
        return nn.Sequential(*layers)

    def forward(self, pxo):
        # 输出Encoder-Decoder之后的点云特征
        # pxo = pxo.reshape(500, 7)
        # p0, x0, o0 = pxo[:, 0:3].contiguous(), pxo[:, 3:6].contiguous(), pxo[:, 6].contiguous()  # (n, 3), (n, c), (b)
        p0, x0, o0 = pxo
        # print(p0.shape, x0.shape, o0.shape)
        # torch.Size([110327, 3]) torch.Size([110327, 3]) torch.Size([4])
        x0 = p0 if self.c == 3 else torch.cat((p0, x0), 1)  # c=6
        # print(x0.shape)
        o0 = o0.type(torch.int)
        p1, x1, o1 = self.enc1([p0, x0, o0])
        p2, x2, o2 = self.enc2([p1, x1, o1])
        p3, x3, o3 = self.enc3([p2, x2, o2])
        p4, x4, o4 = self.enc4([p3, x3, o3])
        p5, x5, o5 = self.enc5([p4, x4, o4])
        x5 = self.dec5[1:]([p5, self.dec5[0]([p5, x5, o5]), o5])[1]
        x4 = self.dec4[1:]([p4, self.dec4[0]([p4, x4, o4], [p5, x5, o5]), o4])[1]
        x3 = self.dec3[1:]([p3, self.dec3[0]([p3, x3, o3], [p4, x4, o4]), o3])[1]
        x2 = self.dec2[1:]([p2, self.dec2[0]([p2, x2, o2], [p3, x3, o3]), o2])[1]
        x1 = self.dec1[1:]([p1, self.dec1[0]([p1, x1, o1], [p2, x2, o2]), o1])[1]
        # x = self.ln1(x1)
        # x = self.bn1(x)
        # x, _ = self.mbn1(x, torch.tensor(0).cuda())
        # x = self.ac(x)
        # x = self.ln2(x)
        # x = self.bn2(x)
        # x, _ = self.mbn2(x, torch.tensor(0).cuda())
        # x = self.ac(x)
        # x = self.logits1(x1)
        # x_d = self.logits2(x1)
        # y_out_normal, _ = torch.max(x_d, dim=1, keepdim=True)
        #
        # y_normal_dummy = torch.cat([x, y_out_normal], dim=1)
        return x1

    def forward_ood(self, pxo):
        x1 = self.forward(pxo)
        y_in = self.logits1(x1)  # 输出13个
        # y_out = self.logits2(x1)  # 输出1个

        # y_out_normal, _ = torch.max(y_out, dim=1, keepdim=True)

        # y_normal_dummy = torch.cat([y_out, y_in], dim=1)
        return x1, y_in

    def forward_with_gene_feat(self, pxo, fake_feat=None):
        feat = self.forward(pxo)
        # feat = torch.cat([ori_feat, fake_feat], dim=0)
        y_in = self.logits1(feat)  # 输出13个
        y_out_normal = self.logits2(feat)  # 输出dummy number个

        y_out_normal_2, _ = torch.max(y_out_normal, dim=1, keepdim=True)

        y_normal_dummy = torch.cat([y_in, y_out_normal_2], dim=1)
        return feat, y_in, y_normal_dummy

    def forward_EDS(self, pxo, fake_feat):
        ori_feat = self.forward(pxo)
        feat = torch.cat([ori_feat, fake_feat], dim=0)
        # feat = self.classifier(feat)
        feat_in = self.logits1(feat)
        feat_out = self.logits2(feat)
        feat_all = torch.cat([feat_out, feat_in], dim=1)
        # pred = nn.functional.log_softmax(self.logits1(feat), dim=1)  # num_points * num_classes
        # pred = nn.functional.softmax(self.logits1(feat), dim=1)  # num_points * num_classes
        # pred_shape = pred.size()
        # pred = pred.unsqueeze(1).expand(pred_shape[0], self.k, pred_shape[1])
        # prototypes = torch.eye(self.k).cuda() * 3
        # # print(features.shape)
        # # print(prototypes.shape)
        # dists = pred - prototypes  # num_points * num_classes * num_classes
        # dist2mean = -torch.sum(dists ** 2, 2)
        # x = dist2mean.contiguous()  # num_points * num_classes
        return feat_all

    def forward_proto_gan(self, feat, bn_label):
        # feat = self.forward(pxo)
        x = self.ln1(feat)
        # x = self.bn1(x)
        x, _ = self.mbn1(x, bn_label)
        x = self.ac(x)
        x = self.ln2(x)
        x = self.bn2(x)
        # x, _ = self.mbn2(x, bn_label)
        # x = self.ac(x)
        # output = self.logits2(feat, bn_label)
        return x

    # def generate_proto(self, pxo, target):
    #     feat = self.forward(pxo)
    #
    #     proto_expan = self.memory.unsqueeze(0).expand(feat.shape[0], self.memo_dim, feat.shape[1])
    #     feat_expan = feat.unsqueeze(1).expand(feat.shape[0], self.memo_dim, feat.shape[1])  # [num_point, 11, 32]
    #     feat_logit = self.cosine_similarity(feat_expan, proto_expan)  # [num_point, 11, 1]
    #     if self.addressing == 'soft':
    #         feat_weight = nn.functional.softmax(feat_logit, dim=1)  # [num_point, 11, 1]
    #     elif self.addressing == 'hard':
    #         feat_weight = (nn.functional.relu(feat_logit - self.threshold) * feat_logit) / (
    #                 torch.abs(feat_logit - self.threshold) + self.epsilon)
    #         feat_weight = nn.functional.normalize(feat_weight, p=1, dim=1)
    #     # print(self.memory, self.memory.shape, feat_weight.shape)
    #     # print(feat_weight[target == 10])
    #     # plt.style.use('seaborn')
    #     # sns.heatmap(feat_logit[target == 3][:10, :].detach().cpu().numpy())
    #     # plt.savefig('/user/lijianan/point-transformer/exp/s3dis/pointtransformer_proto/tsne/heatmap_3.png')
    #     feat = torch.mm(feat_weight, self.memory)
    #     # print(feat[target == 10].unsqueeze(1).expand(feat[target == 10].shape[0], self.memo_dim,
    #     #                                              feat.shape[1]) - self.memory)
    #     out = self.logits1(feat)
    #     # pdb.set_trace()
    #     return self.memory, feat, feat_weight, out

    # def forward_generator(self, input):
    #     fake_feat = self.generator(input)
    #     return fake_feat
    #
    # def forward_discriminator(self, input):
    #     fake_or_real = self.discriminator(input)
    #     return fake_or_real


def pointtransformer_seg_repro(**kwargs):
    model = PointTransformerSeg(PointTransformerBlock, [2, 3, 4, 6, 3], **kwargs)
    return model


model = pointtransformer_seg_repro(c=3, k=13)
# print(model)
# for name, value in model.named_parameters():
#     print(name)
# model.forward_with_gene(name, name)
