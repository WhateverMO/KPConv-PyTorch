import time
import os
import sys
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
from utils.metrics import smooth_metrics,metrics,fast_confusion
# Custom libs

# Dataset
from plyfile import PlyData, PlyElement

def test_accuracy(test_predictions_path=None, test_groundtruth_path=None, label_to_names=None):
    # labels = ['Gnd', 'Trees', 'Car', 'Truck', 'Wire', 'Fence', 'Poles' , 'Bldngs']
    # ignored_labels = ['Unclassified']
    # label_to_names= {0: 'ground',#LASDU
    #                 1: 'building',
    #                 2: 'tree',
    #                 3: 'low_evg',
    #                 4: 'artifact'}
    print('test accuracy start')
    if label_to_names is None:
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
    # label_values = np.zeros((0,), dtype=np.int32)
    label_values=np.sort([k for k, v in label_to_names.items()])
    classes = len(label_values)
    print(len(label_values))
    # final_labels = labels[:]
    # for each in ignored_labels: final_labels.remove(each)
    # Given below is the path to test output and ground truth files.
    # All files in test output file should be present in ground truth folder for successful execution of this file
    # This Also works in windows
    if test_predictions_path is None:
        test_predictions_path = r'D:\Python\KPConv-PyTorch-master\KPConv-PyTorch-master\test_output\2024-02-05_12-58-12\val_preds'
    if test_groundtruth_path is None:
        test_groundtruth_path = r'D:\Python\KPConv-PyTorch-master\KPConv-PyTorch-master\test_output\2024-02-05_12-58-12\val_preds'
    files_pred = [f for f in os.listdir(test_predictions_path) if f[-4:] == '.ply']
    files_ground = [f for f in os.listdir(test_groundtruth_path) if f[-4:] == '.ply']
    if(all(each in files_ground for each in files_pred )):
        print("All files good")
    else:
        print("Error some files at ",test_predictions_path, "not matching with files at",test_groundtruth_path)
        exit()

    once = False
    total_list_micro = list()
    total_list_macro = list()
    total_list_miou =list()
    acc_list=list()
    Cum = np.zeros((len(label_values), len(label_values)), dtype=np.int32)

    ytrue = []
    ypred=[]
    conf_file = join(test_predictions_path, 'conf.txt')
    for c_i,each_file in enumerate(files_pred):
        print('\n Loading.... ', os.path.join(test_predictions_path, each_file))
        data_pred = PlyData.read(os.path.join(test_predictions_path, each_file))
        data_grtr = PlyData.read(os.path.join(test_groundtruth_path, each_file))
        y_true = data_grtr.elements[0]['class']
        y_pred = data_pred.elements[0]['preds']

        # y_true = data_grtr.elements[0]['label'].astype(np.int32)
        # y_pred = data_pred.elements[0]['pred'].astype(np.int32)
        print(y_true.shape, y_pred.shape)
        ytrue+=[y_true]
        ypred+=[y_pred]
        """ # Uncomment these lines for saving confusion matrix in a color scale in pdf format
        C = confusion_matrix(y_true, y_pred, normalize='pred')
        for l_ind, label_value in enumerate(labels):
            if label_value in ignored_labels:
                C = np.delete(C, l_ind, axis=0)
                C = np.delete(C, l_ind, axis=1)
        if not once:
            Cum = C
        else:
            Cum += C
        plt.imshow(Cum)
        ticks = range(len(final_labels))
        plt.xticks(ticks=ticks,labels=final_labels)
        plt.yticks(ticks=ticks,labels=final_labels)
        if not once: plt.colorbar()
        once = True
        plt.title(" Confusion Matrix ")
        plt.savefig("results/"+each_file[:-4]+'.pdf')
        """
        # F1_score_macro = f1_score(y_true, y_pred, average='micro')
        F1_score_macro = f1_score(y_true, y_pred, average='macro')
        # C = confusion_matrix(y_true, y_pred, normalize='pred')
        C = fast_confusion(y_true.astype(np.int32), y_pred.astype(np.int32),label_values).astype(np.int32)
        print(C.shape)
        Cum+=C

        PRE, REC, F1, IoU, ACC = metrics(C)

        # PRE, REC, F1, IoU, ACC = smooth_metrics(C)
        print('F1',F1.sum()/classes,'ACC',ACC,'AVG.F1',F1.sum()/classes)

        print("macro F1: \t",F1_score_macro)
        print('MIOU: \t',IoU.mean())
        total_list_miou += [IoU.mean()]
        total_list_macro += [F1_score_macro]

        acc_list  += [ACC]
    ytrue1=np.concatenate(ytrue,axis=0).astype(np.int32)
    ypred1=np.concatenate(ypred,axis=0).astype(np.int32)
    D=fast_confusion(ytrue1, ypred1,label_values)
    PRE2, REC2, F12, IoU2, ACC2 = metrics(D)

    np.savetxt(conf_file, D, '%12d')
    # avg_micro = sum(total_list_micro)/len(total_list_micro)

    #
    # avg_macro = sum(total_list_macro)/len(total_list_macro)
    # MIOU=sum(total_list_miou)/len(total_list_miou)
    # ACC=sum(acc_list)/len(acc_list)
    print("F1 score per class: ",F12)
    print("IoU per class: ",IoU2)
    print("ACC per class: ",ACC2)
    print(  "|  Avg  F1(macro) : ", F12.sum()/len(label_values),"|  Avg  miou : ",IoU2.sum()/len(label_values),"|  ACC : ",ACC2)
    print("test accuracy end")
    print()
    
if __name__ == '__main__':
    test_accuracy()