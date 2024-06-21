from tools.plot_convergence import plot_convergence
from tools.plot_conf_mat import plot_conf_mat
from tools.label_to_color import label_to_color
from tools.test_models import test_models
from tools.test_accuracy import test_accuracy
from tools.logger import redirect_stdout
from os.path import join
import json


def auto_func(train_fn,label_to_names,rgb_codes,teacher=False,name='',config_=None):
    dataset_path,config = train_fn(config_)
    max_epoch = str(config.max_epoch)
    log_path = config.saving_path
    log_name = config.saving_path.split('/')[-1]
    dataset_original_path = join(dataset_path,'original_ply')
    config_.plot_name = name
    
    with open(join(log_path,'config.json'), 'w') as f:
        json_config = vars(config_)
        json.dump(json_config, f, indent=4)
        
    redirect_stdout(join(log_path,'log_auto.txt'))
    plot_convergence(log_name,log_path,name)
    # get attack types from label_to_names
    attack_types = [name for name in label_to_names.values()]
    plot_conf_mat(join(log_path,'val_preds_'+max_epoch,'conf.txt'),attack_types,log_path)
    label_to_color(dataset_original_path,join(log_path,'val_preds_'+max_epoch),rgb_codes)
    # plot_convergence(log_name)
    # plot_conf_mat(join(log_path,'teacher_val_preds_'+max_epoch,'conf.txt'))
    if teacher:
        label_to_color(dataset_original_path,join(log_path,'teacher_val_preds_'+max_epoch),rgb_codes)
    test_models(log_path)
    test_accuracy(join('test',log_name,'predictions'),dataset_original_path,label_to_names)
    
    return dataset_path,log_name,name

def auto(train_fn,label_to_names,rgb_codes,teacher=False,name='',config_=None):
    from concurrent.futures import ProcessPoolExecutor
    executor = ProcessPoolExecutor(max_workers=1)
    try:
        dataset_path,log_name,name = executor.submit(auto_func,train_fn,label_to_names,rgb_codes,teacher,name,config_).result()
    finally:
        executor.shutdown(wait=True)
    return dataset_path,log_name,name

rgb_codes_dict ={
    'ISPRS':
        [[0, 0, 255], #ISPRS
        [152, 245, 255],
        [190, 190, 190],
        [255, 99, 71],
        [255, 0, 255],
        [255, 0, 0],
        [255, 255, 0],
        [0,255,0],
        [46,139,87]],
    'LASDU':
        [[192,192,192],  # LASDU #ground
        [0,0,255],#building
        [0,100,0],#tree
        [152,251,152],#low veg
        [255,69,0]],#artifact
        
}

label_to_names_dict = {
    'ISPRS':
        {0: 'powerline',
        1: 'low vegetation',
        2: 'impervious surfaces',
        3: 'car',
        4: 'fence/hedge',
        5: 'roof',
        6: 'facade',
        7: 'shurb',
        8: 'tree'
            },
    'LASDU':
        {0: 'ground',
        1: 'building',
        2: 'tree',
        3: 'low_evg',
        4: 'artifact'
        },
        
}

        # rgb_codes = [[200, 90, 0],
        #             [255, 0, 0],
        #             [255, 0, 255],
        #             [0, 220, 0],
        #             [0, 200, 255]]
        # rgb_codes = [[190,190,190],   #DFC2019
        #              [46,139,87],
        #              [255,97,0],
        #              [0,0,255],
        #              [255,255,0]]
        # rgb_codes = [[178, 203, 47],  # H3D
        #             [183, 179, 170],
        #             [33, 151, 163],
        #             [168, 34, 107],
        #             [255, 122, 89],
        #             [254, 215, 136],
        #             [89, 125, 53],
        #             [0, 128, 65],
        #             [170, 86, 0],
        #             [253, 255, 6],
        #             [128, 0,0]]
        # rgb_codes = random_colors(6)
        