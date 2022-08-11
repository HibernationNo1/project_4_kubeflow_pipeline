mode = "labelme"

pipeline = dict(
    pipeline_name = 'train',
    pipeline_version = "0.1"
)



gs = dict(
    client_secrets = "client_secrets.json",
    recoded_dataset_bucket_name = "pipeline_taeuk4958",
    )

dataset = dict(
    train_file_name = 'train_dataset.json',
    val_file_name = 'val_dataset.json',
    dataset_version = '0.1'
)

train = dict(
    validate = False,
    finetun = True,
    model_version = '0.0.1'
)

cfg = {"pipeline" : pipeline,
       'gs' : gs,
       'dataset' : dataset,
       'train' : train}