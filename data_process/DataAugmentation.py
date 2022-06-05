import os
from ImageDataset import label_map
import random
from pathlib import Path
import shutil

import PIL
from PIL import Image
import torchvision.transforms as T
import torch


def randon_transform(img):
    transforms = torch.nn.Sequential(
        T.ColorJitter(hue=.02, saturation=.02),
        T.RandomHorizontalFlip(),
        T.RandomRotation(20, resample=PIL.Image.BILINEAR),
        T.RandomAdjustSharpness(sharpness_factor=2),
        T.RandomAutocontrast(),
    )
    return transforms(img)


def aug(img_root_dir, single_class_len=500):

    img_root_dir_path = Path(img_root_dir)
    img_root_dir_parent_path = img_root_dir_path.parent.absolute()

    aug_count = 0
    while os.path.exists(os.path.join(img_root_dir_parent_path, f'aug_{aug_count}')):
        aug_count += 1

    os.mkdir(os.path.join(img_root_dir_parent_path, f'aug_{aug_count}'))

    for label, label_name in label_map.items():
        class_original_dir = os.path.join(img_root_dir, label_name)
        class_aug_dir = os.path.join(img_root_dir_parent_path,
                                     f'aug_{aug_count}', label_name)
        os.mkdir(class_aug_dir)

        img_file_names = os.listdir(class_original_dir)
        original_dataset_len = len(img_file_names)

        if original_dataset_len > single_class_len:
            # print(
            #     f'size of class "{label_name}" is larger than {single_class_len}, so shuffle the dataset and cut it out')
            random.shuffle(img_file_names)
            img_file_names = img_file_names[:single_class_len]
            print(
                f'label: {label} with {len(img_file_names)} images(cutted), named as "{label_name}"')

        else:
            average_augmented_times_per_img = int(round(
                single_class_len / original_dataset_len, 0))

            aug_left = single_class_len - original_dataset_len

            img_aug_count = {}
            while aug_left > 0:
                picked_img_file_name = random.choice(img_file_names)
                if img_aug_count.get(picked_img_file_name) == None:
                    img_aug_count[picked_img_file_name] = 1

                while img_aug_count[picked_img_file_name] > average_augmented_times_per_img:
                    picked_img_file_name = random.choice(img_file_names)
                    if img_aug_count.get(picked_img_file_name) == None:
                        img_aug_count[picked_img_file_name] = 1

                orig_img = Image.open(os.path.join(
                    class_original_dir, picked_img_file_name))

                aug_img = randon_transform(orig_img)
                aug_img_count = 0
                while os.path.exists(os.path.join(
                        class_aug_dir, picked_img_file_name.replace('.', f'_aug_{aug_img_count}.'))):
                    aug_img_count += 1
                aug_img.save(os.path.join(
                    class_aug_dir, picked_img_file_name.replace('.', f'_aug_{aug_img_count}.')))

                img_aug_count[picked_img_file_name] += 1
                aug_left -= 1

            print(
                f'label: {label} with {original_dataset_len} images(augmented), named as "{label_name}"')

        for img_file_name in img_file_names:
            shutil.copyfile(os.path.join(class_original_dir, img_file_name), os.path.join(
                class_aug_dir, img_file_name))

        label += 1


aug('/Users/yinnnyou/workspace/ai_face_mask_detector/data/resized',
    single_class_len=1000)
