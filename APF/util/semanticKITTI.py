import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "1"
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import numpy as np
import torch
from torch.utils import data
import yaml
from util.data_util import data_prepare, collate_fn


def absoluteFilePaths(directory):
    for dirpath, _, filenames in os.walk(directory):
        filenames.sort()
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))


# def collate_fn(batch):
#     # 列表或元组前面加星号作用是将列表解开成多个独立的参数后传入函数
#     # 字典前面加两个星号，是将字典的值解开成独立的元素作为形参
#     coord, feat, label = list(zip(*batch))
#     return torch.cat(coord), torch.cat(feat), torch.cat(label)


def get_change_dict(init_class, open_label):
    """
    开集数据的标签设置为0~len(open_label)-1
    """
    label_all = [i for i in range(init_class)]
    close_label = [*(set(label_all) - set(open_label))]

    close_label_new = [i for i in range(len(open_label), len(label_all))]
    open_label_new = [i for i in range(len(open_label))]
    # open_label_new = label_all[-len(open_label):]
    # open_label_new = [255 for _ in range(len(open_label))]

    close_label_change = [*(np.array(close_label_new) - np.array(close_label))]
    open_label_change = [*(np.array(open_label_new) - np.array(open_label))]

    label = close_label + open_label
    label_change = close_label_change + open_label_change
    label_change_dict = dict(zip(label, label_change))

    return label_change_dict


def change_label(data, label_change_dict):
    change_value = np.array([label_change_dict[i] for i in data[:, -1]])
    data[:, -1] = data[:, -1] + change_value
    return data


class SemKITTI(data.Dataset):
    def __init__(self, data_path, imageset='train',
                 return_ref=False, label_mapping="/user/lijianan/point-transformer/config/label_mapping/semantic-kitti.yaml", split='train',
                 voxel_size=0.04, voxel_max=None, transform=None, shuffle_index=False, open_label=[5]):
        self.return_ref = return_ref
        self.split, self.voxel_size, self.transform, self.voxel_max, self.shuffle_index = split, voxel_size, transform, voxel_max, shuffle_index
        self.open_label = open_label
        self.label_change_dict = get_change_dict(init_class=20, open_label=self.open_label)
        with open(label_mapping, 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)
        self.learning_map = semkittiyaml['learning_map']
        self.learning_inv_map = semkittiyaml['learning_map_inv']
        self.color_map = semkittiyaml['color_map']
        self.imageset = imageset
        if imageset == 'train':
            split = semkittiyaml['split']['train']
        elif imageset == 'val':
            split = semkittiyaml['split']['valid']
        elif imageset == 'test':
            split = semkittiyaml['split']['test']
        else:
            raise Exception('Split must be train/val/test')

        self.im_idx = []
        for i_folder in split:
            self.im_idx += absoluteFilePaths('/'.join([data_path, str(i_folder).zfill(2), 'velodyne']))

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.im_idx)

    def __getitem__(self, index):
        raw_data = np.fromfile(self.im_idx[index], dtype=np.float32).reshape((-1, 4))
        if self.imageset == 'test':
            annotated_data = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
        else:
            annotated_data = np.fromfile(self.im_idx[index].replace('velodyne', 'labels')[:-3] + 'label',
                                         dtype=np.uint32).reshape((-1, 1))
            instance_data = annotated_data & 0xFFFF
            # instance_data = annotated_data >> 16
            semantic_data = np.vectorize(self.learning_map.__getitem__)(instance_data).reshape(instance_data.shape[0], -1)
            inver_data = np.vectorize(self.learning_inv_map.__getitem__)(semantic_data).reshape(instance_data.shape[0], -1)
        # color_data = self.to_color(instance_data).reshape(instance_data.shape[0], -1)
        color_data = self.to_color(inver_data).reshape(instance_data.shape[0], -1)
        xyz_data = raw_data[:, :3].reshape(instance_data.shape[0], -1)

        data = np.concatenate([xyz_data, color_data, semantic_data], axis=1)
        # data = change_label(data, self.label_change_dict)
        xyz_data, color_data, semantic_data = data[:, 0:3], data[:, 3:6], data[:, 6]
        print(np.unique(semantic_data))
        semantic_data[semantic_data == self.open_label] = 0
        # print(np.unique(semantic_data))
        xyz_data, color_data, semantic_data = data_prepare(xyz_data, color_data, semantic_data, self.split, self.voxel_size, self.voxel_max, self.transform, self.shuffle_index)
        return xyz_data, color_data, semantic_data

    @staticmethod
    def map(label, mapdict):
        # put label from original values to xentropy
        # or vice-versa, depending on dictionary values
        # make learning map a lookup table
        maxkey = 0
        for key, data in mapdict.items():
            if isinstance(data, list):
                nel = len(data)
            else:
                nel = 1
            if key > maxkey:
                maxkey = key
        # +100 hack making lut bigger just in case there are unknown labels
        if nel > 1:
            lut = np.zeros((maxkey + 100, nel), dtype=np.int32)
        else:
            lut = np.zeros((maxkey + 100), dtype=np.int32)
        for key, data in mapdict.items():
            try:
                lut[key] = data
            except IndexError:
                print("Wrong key ", key)
        # do the mapping
        return lut[label]

    def to_color(self, label):
        # put label in original values
        return SemKITTI.map(label, self.color_map)


if __name__ == '__main__':
    val_data = SemKITTI(split='train', data_path='/user/lijianan/Dataset/SemanticKITTI/dataset/sequences', imageset='train',
                        voxel_size=0.04,
                        voxel_max=800000, transform=None)

    trainloader = torch.utils.data.DataLoader(dataset=val_data,
                                              batch_size=128,
                                              shuffle=False,
                                              num_workers=0,
                                              pin_memory=True,
                                              drop_last=True, collate_fn=collate_fn)
    print(len(trainloader))
    for i, data in enumerate(trainloader):
        print('begin')
        # print(data[0].shape, data[1].shape, data[2].shape)
        # print(torch.unique(data[2]))
        # print(data)
    print('done')
