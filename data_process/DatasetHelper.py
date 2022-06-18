import os
import sys
from matplotlib.pyplot import cla
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset

label_map_old = {
    0: 'cloth_mask',
    1: 'no_face_mask',
    2: 'surgical_mask',
    3: 'n95_mask',
    4: 'mask_worn_incorrectly',
}

# label_map = {
#     0: 'cloth_mask_ca_m',
#     1: 'cloth_mask_ca_fm',
#     2: 'cloth_mask_af_m',
#     3: 'cloth_mask_af_fm',
#     4: 'cloth_mask_as_m',
#     5: 'cloth_mask_as_fm',
#     6: 'cloth_mask_ar_m',
#     7: 'cloth_mask_ar_fm',

#     8: 'no_face_mask_ca_m',
#     9: 'no_face_mask_ca_fm',
#     10: 'no_face_mask_af_m',
#     11: 'no_face_mask_af_fm',
#     12: 'no_face_mask_as_m',
#     13: 'no_face_mask_as_fm',
#     14: 'no_face_mask_ar_m',
#     15: 'no_face_mask_ar_fm',

#     16: 'surgical_mask_ca_m',
#     17: 'surgical_mask_ca_fm',
#     18: 'surgical_mask_af_m',
#     19: 'surgical_mask_af_fm',
#     20: 'surgical_mask_as_m',
#     21: 'surgical_mask_as_fm',
#     22: 'surgical_mask_ar_m',
#     23: 'surgical_mask_ar_fm',

#     24: 'n95_mask_ca_m',
#     25: 'n95_mask_ca_fm',
#     26: 'n95_mask_af_m',
#     27: 'n95_mask_af_fm',
#     28: 'n95_mask_as_m',
#     29: 'n95_mask_as_fm',
#     30: 'n95_mask_ar_m',
#     31: 'n95_mask_ar_fm',

#     32: 'mask_worn_incorrectly_ca_m',
#     33: 'mask_worn_incorrectly_ca_fm',
#     34: 'mask_worn_incorrectly_af_m',
#     35: 'mask_worn_incorrectly_af_fm',
#     36: 'mask_worn_incorrectly_as_m',
#     37: 'mask_worn_incorrectly_as_fm',
#     38: 'mask_worn_incorrectly_ar_m',
#     39: 'mask_worn_incorrectly_ar_fm',
# }

label_map = {
    0: 'cloth_mask_caas_m',
    1: 'cloth_mask_caas_fm',
    2: 'cloth_mask_afar_m',
    3: 'cloth_mask_afar_fm',

    4: 'no_face_mask_caas_m',
    5: 'no_face_mask_caas_fm',
    6: 'no_face_mask_afar_m',
    7: 'no_face_mask_afar_fm',

    8: 'surgical_mask_caas_m',
    9: 'surgical_mask_caas_fm',
    10: 'surgical_mask_afar_m',
    11: 'surgical_mask_afar_fm',

    12: 'n95_mask_caas_m',
    13: 'n95_mask_caas_fm',
    14: 'n95_mask_afar_m',
    15: 'n95_mask_afar_fm',

    16: 'mask_worn_incorrectly_caas_m',
    17: 'mask_worn_incorrectly_caas_fm',
    18: 'mask_worn_incorrectly_afar_m',
    19: 'mask_worn_incorrectly_afar_fm',
}

# label_map = {
#     0: 'cloth_mask_ca_m',
#     1: 'cloth_mask_ca_fm',
#     2: 'cloth_mask_asar_m',
#     3: 'cloth_mask_asar_fm',
#     4: 'cloth_mask_af_m',
#     5: 'cloth_mask_af_fm',

#     6: 'no_face_mask_ca_m',
#     7: 'no_face_mask_ca_fm',
#     8: 'no_face_mask_asar_m',
#     9: 'no_face_mask_asar_fm',
#     10: 'no_face_mask_af_m',
#     11: 'no_face_mask_af_fm',

#     12: 'surgical_mask_ca_m',
#     13: 'surgical_mask_ca_fm',
#     14: 'surgical_mask_asar_m',
#     15: 'surgical_mask_asar_fm',
#     16: 'surgical_mask_af_m',
#     17: 'surgical_mask_af_fm',

#     18: 'n95_mask_ca_m',
#     19: 'n95_mask_ca_fm',
#     20: 'n95_mask_asar_m',
#     21: 'n95_mask_asar_fm',
#     22: 'n95_mask_af_m',
#     23: 'n95_mask_af_fm',

#     24: 'mask_worn_incorrectly_ca_m',
#     25: 'mask_worn_incorrectly_ca_fm',
#     26: 'mask_worn_incorrectly_asar_m',
#     27: 'mask_worn_incorrectly_asar_fm',
#     28: 'mask_worn_incorrectly_af_m',
#     29: 'mask_worn_incorrectly_af_fm',
# }


class ImageDataset(Dataset):

    def __init__(self, img_root_dir, transform=None, target_transform=None):
        # self.img_labels = pd.read_csv(annotations_file)
        label = 0
        all_img_file_names = []
        all_img_labels = []

        # for label, label_name in label_map_old.items():
        #     class_dir_ls = [img for img in os.listdir(os.path.join(
        #         img_root_dir, label_name)) if img.endswith('.jpeg')]

        #     labels = [label] * len(class_dir_ls)

        #     all_img_file_names.extend(class_dir_ls)
        #     all_img_labels.extend(labels)

        #     print(
        #         f'label: {label} with {len(class_dir_ls)} images, named as "{label_name}"')
        #     label += 1

        sta_overall = {}
        sta_gen = {}
        sta_race = {}

        for label, first_class_label_name in label_map_old.items():
            class_dir_ls = [img for img in os.listdir(os.path.join(
                img_root_dir, first_class_label_name)) if img.endswith('.jpeg')]

            for image_fila_name in class_dir_ls:
                str_array = list(image_fila_name)
                first_class_label = int(str_array[0])
                gender_class_label = int(str_array[8])
                # merge races group
                race_class_label = 1 if int(str_array[10]) in [1, 3] else 0
                # race_class_label = 2 if int(str_array[10]) == 1 else (
                # 0 if int(str_array[10]) == 0 else 1)

                # print(image_fila_name)
                # print(first_class_label, gender_class_label, race_class_label)
                # print(label_map[(first_class_label * 4) +
                #       gender_class_label + (race_class_label * 2)])
                # print()
                all_img_file_names.append(image_fila_name)
                new_label_number = (first_class_label * 4) + \
                    gender_class_label + (race_class_label * 2)
                all_img_labels.append(new_label_number)

                if sta_overall.get(label_map[new_label_number]) == None:
                    sta_overall[label_map[new_label_number]] = 0
                sta_overall[label_map[new_label_number]] += 1

                gender_class_name = 'male' if gender_class_label == 0 else 'female'
                racegroup_class_name = 'caas' if race_class_label == 0 else 'afar'
                # racegroup_class_name = 'ca' if race_class_label == 0 else (
                #     'asar' if race_class_label == 1 else 'af')

                if sta_gen.get(gender_class_name) == None:
                    sta_gen[gender_class_name] = 0
                sta_gen[gender_class_name] += 1

                if sta_race.get(racegroup_class_name) == None:
                    sta_race[racegroup_class_name] = 0
                sta_race[racegroup_class_name] += 1

        # for k in sta_overall.keys():
        #     print(k, sta_overall[k])

        print('statistic of gender:')
        for k in sta_gen.keys():
            print(k, sta_gen[k])

        print('statistic of race group:')
        for k in sta_race.keys():
            print(k, sta_race[k])

        df = pd.DataFrame(
            data={'img_file_name': all_img_file_names, 'label': all_img_labels})

        self.img_labels = df
        self.img_root_dir = img_root_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        label = self.img_labels.iloc[idx, 1]

        # image_class_dir = label_map[label]
        image_class_dir = '_'.join(label_map[label].split('_')[:-2])

        img_path = os.path.join(
            self.img_root_dir, image_class_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
