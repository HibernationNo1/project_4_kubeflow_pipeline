
train_data_v = 'v0.0.5'

gs = dict(
    download = True,    # if True: download from gs gtorage
    upload = True,      # if True: upload to gs storage
    result_bucket = "result_hibernation",
    train = dict(
        dir = 'train',
        path = train_data_v,		# upload path
        accept_formmat = ['.pth'],  # ".jpg", ".log"
        model = dict(
            count = 10,
            min_epoch = 7,
            important= 'best_model.pth'
        )   
    ),
    eval = dict(
        # download_dir = 'models',
        target = train_data_v,		# download path
        dir = "evaluation",	
        path = train_data_v			# upload path
    ),
    test = dict(
        # download_dir = 'models',
        dir = 'test',
        path = None
    )
)

path = dict(
    component_volume = "/pvc",                # pvc volume path
    work_space = '/workspace',              # path of workspace in docker container
    # local_volume = 'pvc-529780dc-9c00-447c-a8f3-003c3f69d1af_pipeline_hibernation-project-r2m2s-pipeline-pvc-2',
    docker_volume = 'docker_volume' 
)


git = dict(
    remote = "origin",
    branch = dict(dataset_repo = 'master'),		# for run `git commit`.
    dataset = dict(
        repo = "pipeline_dataset",
        ann = dict(
            tag = "board_dataset_ann_v0.0.7"		# should match the git repo tag for download ann_dataset
        ),
        train = dict(
            tag = f"board_dataset_train_{train_data_v}"		# should match the git repo tag for download train_dataset
        )
    ),
    package_repo = "sub_module",
)