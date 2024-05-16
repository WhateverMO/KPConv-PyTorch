from train_ISPRS_weak import train_ISPRS_weak, ISPRSConfig
from tools.auto import auto, rgb_codes_dict, label_to_names_dict

if __name__ == '__main__':
    rgb_codes = rgb_codes_dict['ISPRS']
    label_to_names= label_to_names_dict['ISPRS']
    auto(train_ISPRS_weak,label_to_names,rgb_codes,True,config_==ISPRSConfig())