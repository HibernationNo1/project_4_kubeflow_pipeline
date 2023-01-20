


gs = dict(
    models_bucket = 'models_taeuk4958',
    path = None
    
)

path = dict(
    volume = "/pvc",                # pvc volume path
    work_space = '/workspace',      # path if workspace in docker container
    local_volume = '/opt/local-path-provisioner'\
                   '/pvc-e242aba4-a270-4000-a361-064f7c3dc91e_project-pipeline_hibernation-project-gdcr5-pipeline-pvc-2'
)


git_repo = dict(dataset = "pipeline_dataset",
                package = "hibernation_no1")