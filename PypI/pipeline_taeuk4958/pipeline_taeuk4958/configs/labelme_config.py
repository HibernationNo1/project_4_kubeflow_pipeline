from __future__ import annotations
import datetime
today = datetime.date.today()


mode = "labelme"

dir_info = dict(
    visul_gt_image_dir = 'gt_images'
)

options = dict(
    proportion_val = 0.01,      
    save_gt_image = False,
    only_val_obj = False        # valid_objec에 포함되지 않는 라벨이 있을 때 무시하는 경우 False, Error 발생시키는 경우 True
)

json = dict(
    category = None,
    valid_object = ["leaf", 'midrid', 'stem', 'petiole', 'flower', 'fruit', 'y_fruit', 'cap', 
                    'first_midrid', 'last_midrid', 'mid_midrid', 'side_midrid'],
    train_file_name = 'train_dataset.json',
    val_file_name = 'val_dataset.json'
    )

dataset = dict(
    info = dict(description = 'Hibernation Custom Dataset',
                url = ' ',
                version = '0.0.1',
                year = f"{today.year}",
                contributor = ' ',
                data_created = (f"{today.year}/{today.month}/{today.day}"),
                licenses = dict(url = ' ', id = 1, name = ' ')  
                ) 
    
                )

cfg = {'mode' : mode,
        'dir_info' : dir_info,
        'options' : options,
        'json' : json,
        'dataset' : dataset}