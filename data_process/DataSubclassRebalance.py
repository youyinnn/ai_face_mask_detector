import sys
import os
from pathlib import Path
import numpy as np

import PIL
from PIL import Image
import random
from DataAugmentation import randon_transform

def get_states(file_list):
    original_img_file_list = [f for f in file_list if 'aug' not in f]
    aug_img_file_list = [f for f in file_list if 'aug' in f]
    
    original_img_aug_count = {}
    for original_img in original_img_file_list:
        original_img_aug_count[original_img.replace('.jpeg', '')] = 0
    
    for aug_img in aug_img_file_list:
        original = aug_img[:11]
        original_img_aug_count[original] += 1
        
    subclass_statistic = {
        'm': 0,
        'fm': 0,
        'caas': 0,
        'afar': 0,
    }
        
    for f in file_list:
        str_array = list(f)
        gender_class_label = 'm' if int(str_array[8]) == 0 else 'fm'
        race_class_label = 'caas' if (1 if int(str_array[10]) in [1, 3] else 0) == 0 else 'afar'
        subclass_statistic[gender_class_label] += 1
        subclass_statistic[race_class_label] += 1

    
    return original_img_file_list, aug_img_file_list, original_img_aug_count, subclass_statistic,
    

def rebalance(img_dir_path, f_t, r_t):
    
    img_dir_abs_path = Path(img_dir_path).absolute()

    img_file_paths = ([f for f in os.listdir(img_dir_abs_path) if f.endswith('.jpeg')])
    
    original_img_file_list, aug_img_file_list, original_img_aug_count, subclass_statistic = get_states(img_file_paths)
    
    m_count = subclass_statistic['m']
    fm_count = subclass_statistic['fm']
    caas_count = subclass_statistic['caas']
    afar_count = subclass_statistic['afar']
    
    while abs(m_count - fm_count) > f_t or abs(caas_count - afar_count) > r_t:
        print(f'before: {subclass_statistic}, abs gen: {abs(m_count - fm_count)}, abs race: {abs(caas_count - afar_count)}')
        
        gen_is_urgent = abs(m_count - fm_count) > f_t
        race_is_urgent = abs(caas_count - afar_count) > r_t
        
        should_add_gen = (0 if m_count < fm_count else 1) if gen_is_urgent else random.choice([0, 1])
        should_add_race = ([0, 2] if caas_count < afar_count else [1, 3]) if race_is_urgent else random.choice([[0, 2], [1, 3]])
        
        should_remove_gen = 0 if should_add_gen == 1 else 1
        should_remove_race = [1, 3] if np.sum(should_add_race) == 2 else [0, 2]
        
        add, remove = pick_add_and_remove(should_add_gen, should_add_race, 
                                          should_remove_gen, should_remove_race, 
                                          original_img_aug_count)
        
        print('m' if should_add_gen == 0 else 'fm', 'caas' if np.sum(should_add_race) == 2 else 'afar')
        print('m' if should_remove_gen == 0 else 'fm', 'caas' if np.sum(should_remove_race) == 2 else 'afar')
        print(add, remove)
        print(original_img_aug_count[add], original_img_aug_count[remove])
        
        picked_add_img_file_name = f"{add}.jpeg"
        orig_img = Image.open(os.path.join(
                    img_dir_abs_path, picked_add_img_file_name))
        orig_img = orig_img.convert('RGB')

        aug_img = randon_transform(orig_img)
        aug_img_count = 0
        while os.path.exists(os.path.join(
                img_dir_abs_path, picked_add_img_file_name.replace('.', f'_aug_{aug_img_count}.'))):
            aug_img_count += 1

        aug_img.save(os.path.join(
            img_dir_abs_path, picked_add_img_file_name.replace('.', f'_aug_{aug_img_count}.')))
        
        picked_remove_img_original_file_name = f"{remove}.jpeg"
        picked_remove_img_file_name = None
        aug_img_count = np.max(list(original_img_aug_count.values()))
        while picked_remove_img_file_name == None and aug_img_count >= 0:
            possible_remove_img_path = os.path.join(
                img_dir_abs_path, picked_remove_img_original_file_name.replace('.', f'_aug_{aug_img_count}.'))
            if os.path.exists(possible_remove_img_path):
                picked_remove_img_file_name = possible_remove_img_path
            aug_img_count -= 1

            
        os.remove(picked_remove_img_file_name)
        img_file_paths = sorted([f for f in os.listdir(img_dir_abs_path) if f.endswith('.jpeg')])
    
        _, _, original_img_aug_count, subclass_statistic = get_states(img_file_paths)
        
        m_count = subclass_statistic['m']
        fm_count = subclass_statistic['fm']
        caas_count = subclass_statistic['caas']
        afar_count = subclass_statistic['afar']
        
        print(f'after: {subclass_statistic}, abs gen: {abs(m_count - fm_count)}, abs race: {abs(caas_count - afar_count)}')
         
def pick_add_and_remove(should_add_gen, should_add_race, should_remove_gen, should_remove_race, original_img_aug_count):
    add, remove = None, None
    
    l = list(original_img_aug_count.keys())
    
    random.shuffle(l)
    
    # find if there is suitable image that has not been augmented
    aug_count = 0
    while add == None:
        for original_file in l:
            count = original_img_aug_count[original_file]
            
            str_array = list(original_file)
            gender_class_label = int(str_array[8])
            race_class_label = 1 if int(str_array[10]) in [1, 3] else 0
            
            # can be the add
            if (gender_class_label == should_add_gen and race_class_label in should_add_race):
                if count == aug_count and add == None:
                    add = original_file
                    
            if add != None:
                break
        aug_count += 1

    aug_count = np.max(list(original_img_aug_count.values()))
    # find if there is suitable image that has been augmented
    while remove == None:
        for original_file in l:
            count = original_img_aug_count[original_file]
            
            str_array = list(original_file)
            gender_class_label = int(str_array[8])
            race_class_label = 1 if int(str_array[10]) in [1, 3] else 0
            
            # can be the remove
            if (gender_class_label == should_remove_gen and race_class_label in should_remove_race):
                if count == aug_count and remove == None:
                    remove = original_file
            if remove != None:
                break
        aug_count -= 1
        
    return add, remove

if __name__ == "__main__":
    rebalance(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))