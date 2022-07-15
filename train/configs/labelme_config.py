from __future__ import annotations
import time                      

mode = "labelme"

dir_info = dict(
    labelme_dir = "labelme",
    annotations_dir = "annotations",
    dataset_dir = "train_dataset",      
    train_images_dir = 'trian_images',
    val_images_dir = 'val_images'
)

options = dict(
    ratio_val = 0.01,
    save_gt_image = False,
    visul_gt_image_dir = 'gt_images',
    only_val_obj = False        # valid_objec에 포함되지 않는 라벨이 있을 때 무시하는 경우 False, Error 발생시키는 경우 True
)

json = dict(
    category = None,
    valid_categorys=['paprika', "strawberry", "melon", 'onion', "seeding_pepper", 'cucumber', 'tomato', 'test'],
    valid_object = ["leaf", 'midrid', 'stem', 'petiole', 'flower', 'fruit', 'y_fruit', 'cap', 
                    'first_midrid', 'last_midrid', 'mid_midrid', 'side_midrid'],
    train_file_name = 'train_dataset.json',
    val_file_name = 'val_dataset.json'
    )

dataset = dict(
    info = dict(description = 'Hibernation Custom Dataset',
                url = ' ',
                version = '0.0.1',
                year = 2022,
                contributor = ' ',
                data_created = f"{time.strftime('%Y/%m/%d', time.localtime(time.time()))}"),
                licenses = dict(url = ' ',
                    id = 1,
                    name = ' ')   
    
)