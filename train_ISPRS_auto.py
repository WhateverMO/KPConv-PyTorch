from train_ISPRS import train_ISPRS, ISPRSConfig
from tools.auto import auto,rgb_codes_dict,label_to_names_dict

if __name__ == '__main__':
    rgb_codes = rgb_codes_dict['ISPRS']
    label_to_names= label_to_names_dict['ISPRS']
    auto(train_ISPRS,label_to_names,rgb_codes,False, config_=ISPRSConfig())