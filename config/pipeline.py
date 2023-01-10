kbf = dict(
    dashboard = dict(
        user_n = "winter4958@gmail.com",
        name_space = "project-pipeline",
        host = "http://localhost:8080",
        pw = None
        ),
    pipeline = dict(
        name = "recode",
        pac = "project.yaml",
        discription = "project",
        run_uploaded_pipeline = True,      
        version = None
        ),
    experiment = dict(
        name = "recode"
        ),
    run = dict(
        name = "project"
        ),
    volume = dict(
        share_memory = dict(
            name = "shm",
            medium = "Memory",
            path = '/dev/shm'
        ),
        pvc = dict(
            name = "hibernation_pvc_2",
            resource_name = "pipeline-pvc_2",
            mode = 'VOLUME_MODE_RWO',      # choices in ['VOLUME_MODE_RWO', 'VOLUME_MODE_RWM', 'VOLUME_MODE_ROM']
            size = "10Gi",
            storage_class = None,
            mount_path = "/pvc"
        )

    )
    
    )

secrets = dict(
    gs = dict(
        type = None, 
        project_id = None,
        private_key_id = None,
        private_key = None,
        client_email = None,
        client_id = None,
        auth_uri = None,
        token_uri = None,
        auth_provider_x509_cert_url = None,
        client_x509_cert_url = None   
        ),
    db = dict(
        host = None,
        password = None,
        port = None
        )
    )

