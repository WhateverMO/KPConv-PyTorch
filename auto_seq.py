from tools.auto import auto, rgb_codes_dict, label_to_names_dict
from train_ISPRS import train_ISPRS, ISPRSConfig
from train_ISPRS_weak import train_ISPRS_weak, ISPRSConfig as ISPRSConfig_weak
from tools.plot_convergence import plot_convergence,plot_convergence0
import itertools
from os.path import join,exists
from os import makedirs

compare_name = 'ablation2'

if __name__ == '__main__':
    dataset_name = 'ISPRS'
    i = 0
    names = []
    start = ''
    end = ''
    
    ALL,MT,ER,CC,PL,GC = False,True,False,False,False,False
    # 排列组合所有情况
    elements = [True,False]
    combinations = list(itertools.product(elements, repeat=2))
    
    # check compare_name exist
    if exists(compare_name):
        raise ValueError('compare_name:'+compare_name+' already exist')
    else:
        makedirs(join('pic',compare_name))
    
    for c in combinations:
        if i == 2:
            break
        config_ = ISPRSConfig_weak()
        rgb_codes = rgb_codes_dict[dataset_name]
        label_to_names = label_to_names_dict[dataset_name]
        plt_name = ''
        ER,CC = c
        
        config_.ALL,config_.MT,config_.ER,config_.CC,config_.PL,config_.GC = ALL,MT,ER,CC,PL,GC
        if MT:
            plt_name += 'MT_'
        if ER:
            plt_name += 'ER_'
        if CC:
            plt_name += 'CC_'
        if PL:
            plt_name += 'PL_'
        if GC:
            plt_name += 'GC_'
        plt_name += 'weak'
        
        dataset_path,log_name,name = auto(train_ISPRS_weak, label_to_names, rgb_codes, True, name=plt_name, config_=config_)
        names.append(plt_name)
        if i == 0:
            start = log_name
        i += 1

    config_ = ISPRSConfig()
    rgb_codes = rgb_codes_dict[dataset_name]
    label_to_names = label_to_names_dict[dataset_name]
    plt_name = 'full supervision'
    dataset_path,log_name,name = auto(train_ISPRS, label_to_names, rgb_codes, False, name=plt_name, config_=config_)
    
    names.append(plt_name)
    end = log_name
    
    print('start:',start,'end:',end,'names:',names)

    plot_convergence0(start=start, end=end, names=names,name=compare_name)