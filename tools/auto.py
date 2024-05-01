from tools.plot_convergence import plot_convergence
from tools.plot_conf_mat import plot_conf_mat
from tools.label_to_color import label_to_color
from tools.test_models import test_models
from tools.test_accuracy import test_accuracy
from tools.logger import redirect_stdout
from os.path import join

def auto(train_fn,label_to_names,rgb_codes,teacher=False):
    dataset_path,config = train_fn()
    max_epoch = str(config.max_epoch)
    log_path = config.saving_path
    log_name = config.saving_path.split('/')[-1]
    dataset_original_path = join(dataset_path,'original_ply')
    redirect_stdout(join(log_path,'log_auto.txt'))
    plot_convergence(log_name)
    # get attack types from label_to_names
    attack_types = [name for name in label_to_names.values()]
    plot_conf_mat(join(log_path,'val_preds_'+max_epoch,'conf.txt'),attack_types)
    label_to_color(dataset_original_path,join(log_path,'val_preds_'+max_epoch),rgb_codes)
    # plot_convergence(log_name)
    # plot_conf_mat(join(log_path,'teacher_val_preds_'+max_epoch,'conf.txt'))
    if teacher:
        label_to_color(dataset_original_path,join(log_path,'teacher_val_preds_'+max_epoch))
    test_models(log_path)
    test_accuracy(join('test',log_name,'predictions'),dataset_original_path,label_to_names)
    
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
        # rgb_codes = [[192,192,192],  # LASDU #ground
        #              [0,0,255],#building
        #              [0,100,0],#tree
        #              [152,251,152],#low veg
        #              [255,69,0]]#artifact
        # rgb_codes =[[0, 0, 255], #ISPRS
        #             [152, 245, 255],
        #             [190, 190, 190],
        #             [255, 99, 71],
        #             [255, 0, 255],
        #             [255, 0, 0],
        #             [255, 255, 0],
        #             [0,255,0],
        #             [46,139,87]]
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