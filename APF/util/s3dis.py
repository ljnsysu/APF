import os

import numpy as np
import SharedArray as SA
from torch.utils.data import Dataset

from util.data_util import sa_create
from util.data_util import data_prepare
from util import transform as t
import torch
from util.data_util import collate_fn


class S3DIS(Dataset):
    def __init__(self, split='train', data_root='trainval', test_area=5, voxel_size=0.04, voxel_max=None, transform=None, shuffle_index=False,
                 loop=1):
        super().__init__()
        self.split, self.voxel_size, self.transform, self.voxel_max, self.shuffle_index, self.loop = split, voxel_size, transform, voxel_max, shuffle_index, loop
        # 0:ceiling 1:floor 2:wall 3:beam 4:column 5:window 6:door 7:table 8:chair 9:sofa 10:bookcase 11:board 12:clutter
        # self.open_label = [5, 8, 11]
        data_list = sorted(os.listdir(data_root))
        data_list = [item for item in data_list if 'Area_' in item]
        if split == 'train':
            self.data_list = [item for item in data_list if not 'Area_{}'.format(test_area) in item]
        else:
            self.data_list = [item for item in data_list if 'Area_{}'.format(test_area) in item]
        for item in self.data_list:
            if not os.path.exists("/dev/shm/{}".format(item)):
                data_path = os.path.join(data_root, item)
                data = np.load(data_path)  # xyzrgbl, N*7
                sa_create("shm://{}".format(item), data)
        self.data_idx = np.arange(len(self.data_list))
        print("Totally {} samples in {} set.".format(len(self.data_idx), split))

    def __getitem__(self, idx):
        # idx = rang(batch_size)
        data_idx = self.data_idx[idx % len(self.data_idx)]
        data = SA.attach("shm://{}".format(self.data_list[data_idx])).copy()
        coord, feat, label = data[:, 0:3], data[:, 3:6], data[:, 6]
        coord, feat, label = data_prepare(coord, feat, label, self.split, self.voxel_size, self.voxel_max, self.transform, self.shuffle_index)
        return coord, feat, label

    def __len__(self):
        return len(self.data_idx) * self.loop


# if __name__ == '__main__':
#     train_transform = t.Compose([t.RandomScale([0.9, 1.1]), t.ChromaticAutoContrast(), t.ChromaticTranslation(),
#                                  t.ChromaticJitter(), t.HueSaturationTranslation()])
#     train_data = S3DIS(split='train', data_root='/user/lijianan/point-transformer/data/stanford_indoor3d/', test_area=5,
#                        voxel_size=0.04,
#                        voxel_max=80000, transform=train_transform, shuffle_index=True, loop=30)
#     train_sampler = None
#     train_loader = torch.utils.data.DataLoader(train_data, batch_size=8, shuffle=False,
#                                                num_workers=0, pin_memory=True, sampler=train_sampler,
#                                                drop_last=True, collate_fn=collate_fn)
#     # collect_fn可以自定义取出一个batch数据的格式
#     for i, data in enumerate(train_loader):
#         print(data)
#         print('done')
#     print('done')
