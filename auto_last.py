from tools.auto import auto
import numpy as np
import os
from utils.config import Config

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
    rgb_codes =[[0, 0, 255], #ISPRS
                [152, 245, 255],
                [190, 190, 190],
                [255, 99, 71],
                [255, 0, 255],
                [255, 0, 0],
                [255, 255, 0],
                [0,255,0],
                [46,139,87]]
    label_to_names= {0: 'powerline',
                            1: 'low vegetation',
                            2: 'impervious surfaces',
                            3: 'car',
                            4: 'fence/hedge',
                            5: 'roof',
                            6: 'facade',
                            7: 'shurb',
                            8: 'tree'
                                }
    auto(get_last,label_to_names,rgb_codes,True)