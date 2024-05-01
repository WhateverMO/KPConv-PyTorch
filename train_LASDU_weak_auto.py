from train_LASDU_weak import train_LASDU_weak
from tools.auto import auto

if __name__ == '__main__':
    rgb_codes = [[192,192,192],  # LASDU #ground
                     [0,0,255],#building
                     [0,100,0],#tree
                     [152,251,152],#low veg
                     [255,69,0]]#artifact
    label_to_names= {0: 'ground',
                    1: 'building',
                    2: 'tree',
                    3: 'low_evg',
                    4: 'artifact'}
    auto(train_LASDU_weak,label_to_names,rgb_codes,True)