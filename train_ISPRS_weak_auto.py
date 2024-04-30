from train_ISPRS_weak import train_ISPRS_weak
from tools.plot_convergence import plot_convergence
from tools.plot_conf_mat import plot_conf_mat
from tools.label_to_color import label_to_color
from tools.test_models import test_models
from tools.test_accuracy import test_accuracy
from tools.logger import redirect_stdout
from os.path import join

if __name__ == '__main__':
    dataset_path,config = train_ISPRS_weak()
    max_epoch = str(config.max_epoch)
    log_path = config.saving_path
    log_name = config.saving_path.split('/')[-1]
    dataset_original_path = join(dataset_path,'original_ply')
    redirect_stdout(join(log_path,'log_auto.txt'))
    plot_convergence(log_name)
    attack_types = ['Powerline', 'Low vegetation', 'surface', 'Car', 'Fence', 'Roof', 'Facade', 'shurb', 'Tree',]
    plot_conf_mat(join(log_path,'val_preds_'+max_epoch,'conf.txt'),attack_types)
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
    test_accuracy(join('test',log_name,'predictions'),dataset_original_path,label_to_names)
    test_models(log_path)
    label_to_color(dataset_original_path,join(log_path,'val_preds_'+max_epoch))
    # plot_convergence(log_name)
    # plot_conf_mat(join(log_path,'teacher_val_preds_'+max_epoch,'conf.txt'))
    label_to_color(dataset_original_path,join(log_path,'teacher_val_preds_'+max_epoch))