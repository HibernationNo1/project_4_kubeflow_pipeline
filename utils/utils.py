import json
import numpy as np
import datetime


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, datetime.date):
            return 'datetime.date'
        else:
            return super(NpEncoder, self).default(obj)

def filter_config(from_config, to_config):  
    """
    Exclude all except type: (dict, list, bool, int, loat, str)
    """
    if isinstance(from_config, dict):
        for key in list(from_config.keys()):            
            if isinstance(from_config[key], dict):
                to_config[key] = {}  
                to_config[key] =  filter_config(from_config[key], to_config[key])
            if isinstance(from_config[key], list):
                to_config[key] = []
                to_config[key] = filter_config(from_config[key], to_config[key])
            if isinstance(from_config[key], bool):
                to_config[key] = from_config[key]
            if isinstance(from_config[key], int):
                to_config[key] = from_config[key]
            if isinstance(from_config[key], float):
                to_config[key] = from_config[key]
            if isinstance(from_config[key], str):
                to_config[key] = from_config[key]
    if isinstance(from_config, list):
        for value in from_config:
            if isinstance(value, dict):
                tmp_to_config = {}
                to_config.append(filter_config(value, tmp_to_config))
            if isinstance(value, list):
                tmp_to_config = []
                to_config.append(filter_config(value, tmp_to_config))
            if isinstance(value, bool):
                to_config.append(value)
            if isinstance(value, int):
                to_config.append(value)
            if isinstance(value, float):
                to_config.append(value)
            if isinstance(value, str):
                to_config.append(value)
                
    return to_config