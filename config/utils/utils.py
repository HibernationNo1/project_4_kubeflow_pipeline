


gs = dict(
    download = True,    # if True: download from gs gtorage
    upload = True,      # if True: upload to gs storage
    result_bucket = "result_hibernation",
    train = dict(
        dir = 'train',
        path = '0.0.2',
        accept_formmat = ['.pth'],  # ".jpg", ".log"
        model = dict(
            count = 10,
            min_epoch = 7,
            important= 'best_model.pth'
        )   
    ),
    eval = dict(
        # download_dir = 'models',
        target = "0.0.2",
        dir = "evaluation",
        path = "0.0.2"
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
    branch = dict(dataset_repo = 'develop'),		# for run `git commit`.
    dataset = dict(
        repo = "pipeline_dataset",
        ann = dict(
            tag = "board_dataset_v0.0.5",
            version = "0.0.5"
        ),
        train = dict(
            tag = "board_dataset_train_0.0.1",
            version = "0.0.1"
        )
    ),
    package_repo = "sub_module",
)