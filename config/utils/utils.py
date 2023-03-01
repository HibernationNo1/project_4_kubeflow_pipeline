


gs = dict(
    models_bucket = 'models_taeuk4958',
    path = None
    
)

path = dict(
    volume = "/pvc",                # pvc volume path
    work_space = '/workspace',      # path if workspace in docker container
    local_volume = '/opt/local-path-provisioner'\
                   '/pvc-c5ecf260-b80c-4cf4-92ff-e5a7bf57aac1_pipeline_hibernation-project-pggtm-pipeline-pvc-2' 
)


git_repo = dict(dataset = "pipeline_dataset",
                package = "hibernation_no1")