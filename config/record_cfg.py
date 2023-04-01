_base_ = [
    "utils/dvc.py",
    "utils/database.py",
    "utils/utils.py"
]


ann_data_root = 'ann_dataset'
record_result = "test_dataset"

record = dict(
    options = dict(
    proportion_val = 0.1
    # save_gt_image = False
    ),
    
    info = dict(description = 'Hibernation Custom Dataset',
                url = ' ',
                version = '0.0.1',
                contributor = ' ',
                licenses = dict(url = ' ', id = 1, name = ' ')  
                ), 
    category = None,
    valid_object = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0',
                    'a', 'b', 'c', 'd', 'e',
                    "r_board", "r_m_n", "r_s_n", "l_board", "l_m_n", "l_s_n"],
    train_dataset = 'train_dataset.json',
    val_dataset = 'val_dataset.json'
                )

