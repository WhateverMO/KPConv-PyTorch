from train_LASDU_weak import train_LASDU_weak, LASDUConfig
from tools.auto import auto, rgb_codes_dict, label_to_names_dict

if __name__ == '__main__':
    rgb_codes = rgb_codes_dict['LASDU']
    label_to_names= label_to_names_dict['LASDU']
    auto(train_LASDU_weak,label_to_names,rgb_codes,True, config_=LASDUConfig())