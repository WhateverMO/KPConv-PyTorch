# def read_pts_file(file_path,lable = True):
#     with open(file_path, 'r') as f:
#         lines = f.readlines()

#     points = []
#     lables = []
#     for line in lines:
#         parts = line.split()
#         point = [float(it) for it in parts[:3]]
#         points.append(point)
#         if lable:
#             lables.append(int(parts[-1]))
#     return points,lables

# points,lables = read_pts_file('../../Data/ISPRS3D/Vaihingen3D_Traininig.pts')
# input('done!')

# from collections import Counter

# def calculate_label_distribution(labels):
#     counter = Counter(labels)
#     total_count = len(labels)
#     distribution = {label: count / total_count for label, count in counter.items()}
#     return distribution

# distribution = calculate_label_distribution(lables)
# for label, percentage in distribution.items():
#     print(f"Label {label}: {percentage * 100:.2f}%")


# import numpy as np

# data = np.array([0,1,2,3,4,5,6,7,8,9,10])
# input_inds = np.array([1,3,5,6,7,8,10])
# weak_inds = np.array([2,3,6,10])

# inner = np.in1d(input_inds, weak_inds)

# origin_data = data[input_inds]
# weak_data = data[weak_inds]
# need_data = origin_data[inner]

# print(inner)
# print(origin_data)
# print(weak_data)
# print(need_data)

# import sys

# class Logger(object):
#     def __init__(self, filename='output.txt'):
#         self.terminal = sys.stdout
#         self.log = open(filename, 'w')

#     def write(self, message):
#         self.terminal.write(message)
#         self.log.write(message)

#     def flush(self):
#         # this flush method is needed for python 3 compatibility.
#         # this handles the flush command by doing nothing.
#         # you might want to specify some extra behavior here.
#         pass

# # Redirect stdout
# sys.stdout = Logger('output.txt')

# # Now, any print statements will write to both the file and the console
# print('This will be written to the file and console')
# print('This will be written to the file and console')
# from os.path import join
# log_path = 'results/log'
# max_epoch = '100'
# path = join(log_path,'val_preds_'+max_epoch,'conf.txt')

# # get the second word of path
# second_word = path.split('/')[1]
# print(second_word)
import numpy as np

a = np.array([2,5,6,5,10])

print(np.where(a == 5))