_base_ = [
    "utils/dvc.py",
    "utils/database.py",
]





recode = dict(
    options = dict(
    proportion_val = 0.01
    # save_gt_image = False
    ),
    
    info = dict(description = 'Hibernation Custom Dataset',
                url = ' ',
                version = '0.0.1',
                contributor = ' ',
                licenses = dict(url = ' ', id = 1, name = ' ')  
                ), 
    category = None,
    valid_object = ["leaf", 'midrid', 'stem', 'petiole', 'flower', 'fruit', 'y_fruit', 'cap', 'cap_2', 
                    'first_midrid', 'last_midrid', 'mid_midrid', 'side_midrid'],
    train_dataset = 'train_dataset.json',
    val_dataset = 'val_dataset.json'
                )

