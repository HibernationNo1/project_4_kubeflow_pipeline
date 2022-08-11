import json
import os

def set_gs_credentials(client_secrets_file_name, gs_secret):
    """
    Args:
        client_secrets_file_name (str): file name (json format)
        gs_secret (dict): google client secret dict
    """
    client_secrets_path = os.path.join(os.getcwd(), client_secrets_file_name)
        
    # save client_secrets
    json.dump(gs_secret, open(client_secrets_path, "w"), indent=4)
    
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = client_secrets_path