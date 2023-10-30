import pandas as pd
from torch.utils import data
import numpy as np
from torchvision import transforms
from PIL import Image, ImageFilter
import torch
import json


def getData(mode):
    with open('./objects.json') as f:
        object_dict = json.load(f)
    if mode == 'train':
        with open('./train.json') as f:
            data_input = json.load(f)
        label = np.zeros((len(data_input), 24))
        img_path = []

        for i, (key, values) in enumerate(data_input.items()):
            img_path.append(key)
            for objects in values:
                label[i][object_dict[objects]] = 1

        return np.array(img_path), label, object_dict
    else:
        with open('./test.json') as f:
            data_input = json.load(f)
        
        label = np.zeros((len(data_input), 24))
        for i, (data) in enumerate(data_input):
            for objects in data:
                label[i][object_dict[objects]] = 1

        return label, object_dict


class ImageLoader(data.Dataset):
    def __init__(self, root, mode, size):
        self.root = root
        self.mode = mode
        if mode == 'train':
            self.img_name, self.label, self.object_dict = getData(mode)
        else:
            self.label, self.object_dict = getData(mode)
        self.transform = transforms.Compose(
                            [transforms.Resize(size, interpolation=Image.BILINEAR),
                            transforms.ToTensor(),
                            transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
                            ])

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        if self.mode == 'train':
            path = self.root + self.img_name[index]
            label = self.label[index]
            img = Image.open(path).convert('RGB')
            img = self.transform(img)

            return img, torch.from_numpy(label.astype(np.float32))
        else:
            label = self.label[index]
            return torch.from_numpy(label.astype(np.float32))


