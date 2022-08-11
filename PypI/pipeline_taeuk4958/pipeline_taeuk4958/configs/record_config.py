
import datetime
today = str(datetime.date.today())


mode = "labelme"

pipeline = dict(
    pipeline_name = 'record',
    pipeline_version = "0.1"
)


options = dict(
    proportion_val = 0.01,      
    save_gt_image = False,
    only_val_obj = False        # valid_objec에 포함되지 않는 라벨이 있을 때 무시하는 경우 False, Error 발생시키는 경우 True
)

gs = dict(
    client_secrets = "client_secrets.json",
    ann_bucket_name = "dataset_tesuk4958",
    recoded_dataset_bucket_name = "pipeline_taeuk4958",
    recoded_dataset_version = "test_0.1"
    )

dataset = dict(
    dataset_dir = "dataset",
    info = dict(description = 'Hibernation Custom Dataset',
                url = ' ',
                version = '0.0.1',
                year = f"{today.split('-')[0]}",
                contributor = ' ',
                data_created = (f"{today.split('-')[0]}/{today.split('-')[1]}/{today.split('-')[2]}"),
                licenses = dict(url = ' ', id = 1, name = ' ')  
                ), 
    category = None,
    valid_object = ["leaf", 'midrid', 'stem', 'petiole', 'flower', 'fruit', 'y_fruit', 'cap', 
                    'first_midrid', 'last_midrid', 'mid_midrid', 'side_midrid'],
    train_file_name = 'train_dataset.json',
    val_file_name = 'val_dataset.json'
                )


cfg = {"pipeline" : pipeline ,
       'mode' : mode,
       'options' : options,
       'gs' : gs,
       'dataset' : dataset}