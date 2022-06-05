import os
import sys
from matplotlib.pyplot import cla
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset

label_map = {
    0: 'cloth_mask',
    1: 'no_face_mask',
    2: 'surgical_mask',
    3: 'n95_mask',
    4: 'mask_worn_incorrectly',
}


class ImageDataset(Dataset):

    def __init__(self, img_root_dir, train=True, transform=None, target_transform=None):
        # self.img_labels = pd.read_csv(annotations_file)
        label = 0
        all_img_file_names = []
        all_img_labels = []

        for label, label_name in label_map.items():
            class_dir_ls = os.listdir(os.path.join(
                img_root_dir, label_name, 'train' if train else 'test'))

            labels = [label] * len(class_dir_ls)

            all_img_file_names.extend(class_dir_ls)
            all_img_labels.extend(labels)

            label_map[label] = label_name

            print(
                f'label: {label} with {len(class_dir_ls)} images, named as "{label_name}"')
            label += 1

        df = pd.DataFrame(
            data={'img_file_name': all_img_file_names, 'label': all_img_labels})

        self.img_labels = df
        self.img_root_dir = img_root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        label = self.img_labels.iloc[idx, 1]
        img_path = os.path.join(
            self.img_root_dir, label_map[label], 'train' if self.train else 'test', self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
