# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 12:16:31 2017

@author: Biagio Brattoli
"""
import os

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

datapath = '/home/pc1/data/torrents/'

trainval = 'train'


# trainval = 'val'


def main():
    # data = DataLoader(datapath+'/ILSVRC2012_img_train', datapath+'/ilsvrc12_train.txt')
    listfile = os.path.join(datapath, 'ilsvrc12_' + trainval + '.txt')
    data = DataLoader(datapath + '/ILSVRC2012_' + trainval, listfile)
    loader = torch.utils.data.DataLoader(dataset=data, batch_size=1,
                                         shuffle=False, num_workers=20)

    count = 0
    for i, filename in enumerate(tqdm(loader)):
        count += 1


class DataLoader(data.Dataset):
    def __init__(self, data_path, txt_list):
        self.data_path = data_path if data_path[-1] != '/' else data_path[:-1]
        self.names = self.__dataset_info(txt_list)
        self.__image_transformer = transforms.Compose([
            transforms.Resize(256, Image.BILINEAR),
            transforms.CenterCrop(255)])
        self.save_path = self.data_path + '_255x255/'
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        for name in self.names:
            if '/' in name:
                fold = self.save_path + name[:name.rfind('/')]
                if not os.path.exists(fold):
                    os.makedirs(fold)

    def __getitem__(self, index):
        name = self.names[index]
        if os.path.exists(self.save_path + name):
            return None, None

        filename = self.data_path + '/' + name
        img = Image.open(filename).convert('RGB')
        img = self.__image_transformer(img)
        img.save(self.save_path + name)
        return self.names[index]

    def __len__(self):
        return len(self.names)

    def __dataset_info(self, txt_labels):
        with open(txt_labels, 'r') as f:
            images_list = f.readlines()

        file_names = []
        for row in images_list:
            file_names.append(row.strip())

        return file_names


if __name__ == "__main__":
    main()
