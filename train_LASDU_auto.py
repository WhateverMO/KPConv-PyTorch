from train_LASDU import train_LASDU
from tools.auto import auto, rgb_codes_dict, label_to_names_dict

if __name__ == '__main__':
    rgb_codes = rgb_codes_dict['LASDU']
    label_to_names= label_to_names_dict['LASDU']
    auto(train_LASDU,label_to_names,rgb_codes,False)