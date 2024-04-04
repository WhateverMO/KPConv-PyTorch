import itertools
import numpy as np
# import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

auto = False

# 绘制混淆矩阵
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues, name='conf'):
    """
,  This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Input
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:显示百分比, False:显示个数
    """
    font = {'family': 'Times New Roman',
            'weight': 'normal',
            'size': 14,
            }

    # 设置输出的图片大小
    figsize = 8, 6
    figure, ax = plt.subplots(figsize=figsize)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.ylabel('True label', font)
    plt.xlabel('Predicted label', font)
    # plt.savefig(name + '.png', dpi=300)
    plt.savefig(auto+'conf.png')
    plt.show()

#
# cnf_matrix_unsu_m2r = np.array([[16, 0, 0, 0, 0, 1, 3, 1, 1, 4],
#                              [1, 27, 4, 0, 3, 2, 5, 24, 4, 15],
#                              [1, 5, 48, 0, 3, 20, 13, 37, 6, 13],
#                              [0, 0, 18, 0, 2, 11, 51, 48, 0, 19],
#                              [3, 37, 16, 0, 399, 123, 96, 100, 4, 23],
#                              [0, 0, 4, 0, 0, 15, 4, 17, 0, 1],
#                              [0, 0, 16, 0, 0, 2, 33, 8, 0, 2],
#                              [0, 0, 0, 0, 0, 0, 1, 24, 0, 0],
#                              [3, 10, 6, 1, 29, 4, 4, 32, 41, 4],
#                              [5, 16, 18, 2, 1, 14, 22, 64, 16, 143]])

#
# attack_types = ['Bathtub', 'Bed', 'Bookshelf', 'Cabinet', 'Chair', 'Lamp', 'Monitor', 'Plant', 'Sofa', 'Table',]

def plot_conf_mat(conf_path = None, attack_types = None):
    print('plot confusion matrix start')
    if conf_path is None:
        path = r'D:\Python\KPConv-PyTorch-master\KPConv-PyTorch-master\test_HSCN\A\conf.txt'
    else:
        global auto
        auto = conf_path.split('/')
        auto = auto[1]+'_'+auto[2]
        path = conf_path
    cnf_matrix_unsu_m2r=np.loadtxt(path)
    if attack_types is None:
        attack_types = ['Powerline', 'Low vegetation', 'surface', 'Car', 'Fence', 'Roof', 'Facade', 'shurb', 'Tree',]

    plot_confusion_matrix(cnf_matrix_unsu_m2r, classes=attack_types, normalize=True, title='Normalized confusion matrix', name='2024_confusion_matrix_unsu_m2r')
    print('plot confusion matrix end')
    print()

if __name__ == '__main__':
    plot_conf_mat()