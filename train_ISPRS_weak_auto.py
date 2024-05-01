from train_ISPRS_weak import train_ISPRS_weak
from tools.auto import auto

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
    auto(train_ISPRS_weak,rgb_codes,label_to_names,True)