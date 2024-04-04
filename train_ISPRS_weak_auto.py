from train_ISPRS_weak import train_ISPRS_weak
from tools.plot_convergence import plot_convergence
from tools.plot_conf_mat import plot_conf_mat
from tools.label_to_color import label_to_color
from tools.test_models import test_models
from tools.test_accuracy import test_accuracy
from os.path import join

if __name__ == '__main__':
    dataset_path,config = train_ISPRS_weak()
    log_path = config.saving_path
    log_name = config.saving_path.split('/')[-1]
    dataset_original_path = join(dataset_path,'original_ply')
    plot_convergence(log_path)
    plot_conf_mat(join(log_path,'val_preds_500','conf.txt'))
    label_to_color(dataset_original_path,join(log_path,'val_preds_500'))
    # plot_convergence(log_path)
    plot_conf_mat(join(log_path,'teacher_val_preds_500','conf.txt'))
    label_to_color(dataset_original_path,join(log_path,'teacher_val_preds_500'))
    test_models(log_path)
    test_accuracy(join('test',log_name,'predictions'),dataset_original_path)