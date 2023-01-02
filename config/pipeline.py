kbf = dict(
    dashboard = dict(
        user_n = "winter4958@gmail.com",
        name_space = "project-pipeline",
        host = "http://localhost:8080",
        pw = None
        ),
    pipeline = dict(
        name = "train",
        pac = "project.yaml",
        discription = "project",
        run_uploaded_pipeline = True,      
        version = None
        ),
    experiment = dict(
        name = "test01"
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
        data = dict(
            name = "data",
            medium = "Memory",
            path = '/home/tmp'
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

