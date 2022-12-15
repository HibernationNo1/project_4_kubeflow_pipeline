
_base_ = [
    "pipeline/dvc.py",
    "pipeline/database.py",
    "pipeline/gs.py"
]





recode = dict(
    options = dict(
    proportion_val = 0.01,      
    save_gt_image = False
    ),
    
    info = dict(description = 'Hibernation Custom Dataset',
                url = ' ',
                version = '0.0.1',
                contributor = ' ',
                licenses = dict(url = ' ', id = 1, name = ' ')  
                ), 
    category = None,
    valid_object = ["leaf", 'midrid', 'stem', 'petiole', 'flower', 'fruit', 'y_fruit', 'cap', 
                    'first_midrid', 'last_midrid', 'mid_midrid', 'side_midrid'],
    train_file_name = 'train_dataset.json',
    val_file_name = 'val_dataset.json'
                )

