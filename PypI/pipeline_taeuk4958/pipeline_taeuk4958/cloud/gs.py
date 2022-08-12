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
    
    
def get_client_secrets():
    client_secrets_dict = {'type':os.environ['type'],
                           'project_id': os.environ['project_id'],
                           'private_key_id': os.environ['private_key_id'],
                           'private_key': os.environ['private_key'],
                           'client_email': os.environ['client_email'],
                           'client_id': os.environ['client_id'],
                           'auth_uri': os.environ['auth_uri'],
                           'token_uri': os.environ['token_uri'],
                           'auth_provider_x509_cert_url': os.environ['auth_provider_x509_cert_url'],
                           'client_x509_cert_url': os.environ['client_x509_cert_url'],
                            }
    

    return client_secrets_dict


def gs_credentials(path = None):
    if path is not None:
        if not path.endswith('.json'): raise TypeError(f"The path : {path} format must be json")
    else:
        path = 'client_secrets.json'
        
    gs_secret = get_client_secrets()
    set_gs_credentials(path, gs_secret)
        