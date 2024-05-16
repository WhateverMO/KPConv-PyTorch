from tools.auto import auto, rgb_codes_dict, label_to_names_dict
import numpy as np
import os
from utils.config import Config

dataset_name = 'ISPRS'

def get_last():
    datasets = {
        'ISPRS':'../../Data/ISPRS3D'
    }
    logs = np.sort([os.path.join('results', f) for f in os.listdir('results') if f.startswith('Log')])

    log = logs[-1]
    log_config = Config()
    log_config.load(log)
    dataset = log_config.dataset
    return datasets[dataset],log_config

if __name__ == '__main__':
    rgb_codes = rgb_codes_dict[dataset_name]
    label_to_names= label_to_names_dict[dataset_name]
    auto(get_last,label_to_names,rgb_codes,True)