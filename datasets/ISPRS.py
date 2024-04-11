#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Class handling ISPRS dataset.
#      Implements a Dataset, a Sampler, and a collate_fn
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 11/06/2018
#


# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#

# Common libs
import time
import numpy as np
import pickle
import torch
import math
import warnings
from multiprocessing import Lock


# OS functions
from os import listdir
from os.path import exists, join, isdir

# Dataset parent class
from datasets.common import PointCloudDataset
from torch.utils.data import Sampler, get_worker_info
from utils.mayavi_visu import *

from datasets.common import grid_subsampling
from utils.config import bcolors


# ----------------------------------------------------------------------------------------------------------------------
#
#           Dataset class definition
#       \******************************/

def read_pts_file(file_path,lable = True):
    data = np.loadtxt(file_path)
    
    points = data[:,0:3]
    features = data[:,3:6]
    lables = np.ones_like(data[:,0])
    if lable:
        lables = data[:,-1]
    return points,features,lables

class ISPRSDataset(PointCloudDataset):
    """Class to handle ISPRS dataset."""

    def __init__(self, config, set='training', use_potentials=True, load_data=True):
        """
        This dataset is small enough to be stored in-memory, so load all point clouds here
        """
        PointCloudDataset.__init__(self, 'ISPRS')

        ############
        # Parameters
        ############

        # Dict from labels to names
        self.label_to_names = {0: 'Powerline',
                               1: 'Low vegetation',
                               2: 'surface',
                               3: 'Car',
                               4: 'Fence',
                               5: 'Roof',
                               6: 'Facade',
                               7: 'shurb',
                               8: 'Tree'
                               }

        # Initialize a bunch of variables concerning class labels
        self.init_labels()

        # List of classes ignored during training (can be empty)
        self.ignored_labels = np.array([])

        # Dataset folder
        self.path = '../../Data/ISPRS3D'

        # Type of task conducted on this dataset
        self.dataset_task = 'cloud_segmentation'

        # Update number of class and data task in configuration
        config.num_classes = self.num_classes - len(self.ignored_labels)
        config.dataset_task = self.dataset_task

        # Parameters from config
        self.config = config

        # Training or test set
        self.set = set

        # Using potential or random epoch generation
        self.use_potentials = use_potentials

        # Path of the training files
        self.train_path = 'original_ply'

        # List of files to process
        ply_path = join(self.path, self.train_path)

        # Proportion of validation scenes
        self.cloud_names = ['Vaihingen3D_Traininig','Vaihingen3D_EVAL_WITH_REF']
        self.all_splits = [0, 1]
        self.validation_split = 1

        # Number of models used per epoch
        if self.set == 'training':
            self.epoch_n = config.epoch_steps * config.batch_num
        elif self.set in ['validation', 'test', 'ERF']:
            self.epoch_n = config.validation_size * config.batch_num
        else:
            raise ValueError('Unknown set for ISPRS data: ', self.set)

        # Stop data is not needed
        if not load_data:
            return

        ###################
        # Prepare ply files
        ###################

        self.prepare_ISPRS_ply()

        ################
        # Load ply files
        ################

        # List of training files
        self.files = []
        for i, f in enumerate(self.cloud_names):
            if self.set == 'training':
                if self.all_splits[i] != self.validation_split:
                    self.files += [join(ply_path, f + '.ply')]
            elif self.set in ['validation', 'test', 'ERF']:
                if self.all_splits[i] == self.validation_split:
                    self.files += [join(ply_path, f + '.ply')]
            else:
                raise ValueError('Unknown set for ISPRS data: ', self.set)

        if self.set == 'training':
            self.cloud_names = [f for i, f in enumerate(self.cloud_names)
                                if self.all_splits[i] != self.validation_split]
        elif self.set in ['validation', 'test', 'ERF']:
            self.cloud_names = [f for i, f in enumerate(self.cloud_names)
                                if self.all_splits[i] == self.validation_split]

        if 0 < self.config.first_subsampling_dl <= 0.01:
            raise ValueError('subsampling_parameter too low (should be over 1 cm')

        # Initiate containers
        self.input_trees = []
        self.input_colors = []
        self.input_labels = []
        self.pot_trees = []
        self.num_clouds = 0
        self.test_proj = []
        self.validation_labels = []
        if self.config.weak_supervision:   
            self.weak_learning_label = config.weak_learning_label
            self.weak_inds = []

        # Start loading
        self.load_subsampled_clouds()

        ############################
        # Batch selection parameters
        ############################

        # Initialize value for batch limit (max number of points per batch).
        self.batch_limit = torch.tensor([1], dtype=torch.float32)
        self.batch_limit.share_memory_()

        # Initialize potentials
        if use_potentials:
            self.potentials = []
            self.min_potentials = []
            self.argmin_potentials = []
            for i, tree in enumerate(self.pot_trees):
                self.potentials += [torch.from_numpy(np.random.rand(tree.data.shape[0]) * 1e-3)]
                min_ind = int(torch.argmin(self.potentials[-1]))
                self.argmin_potentials += [min_ind]
                self.min_potentials += [float(self.potentials[-1][min_ind])]

            # Share potential memory
            self.argmin_potentials = torch.from_numpy(np.array(self.argmin_potentials, dtype=np.int64))
            self.min_potentials = torch.from_numpy(np.array(self.min_potentials, dtype=np.float64))
            self.argmin_potentials.share_memory_()
            self.min_potentials.share_memory_()
            for i, _ in enumerate(self.pot_trees):
                self.potentials[i].share_memory_()

            self.worker_waiting = torch.tensor([0 for _ in range(config.input_threads)], dtype=torch.int32)
            self.worker_waiting.share_memory_()
            self.epoch_inds = None
            self.epoch_i = 0

        else:
            self.potentials = None
            self.min_potentials = None
            self.argmin_potentials = None
            self.epoch_inds = torch.from_numpy(np.zeros((2, self.epoch_n), dtype=np.int64))
            self.epoch_i = torch.from_numpy(np.zeros((1,), dtype=np.int64))
            self.epoch_i.share_memory_()
            self.epoch_inds.share_memory_()

        self.worker_lock = Lock()

        # For ERF visualization, we want only one cloud per batch and no randomness
        if self.set == 'ERF':
            self.batch_limit = torch.tensor([1], dtype=torch.float32)
            self.batch_limit.share_memory_()
            np.random.seed(42)

        return

    def __len__(self):
        """
        Return the length of data here
        """
        return len(self.cloud_names)

    def __getitem__(self, batch_i):
        """
        The main thread gives a list of indices to load a batch. Each worker is going to work in parallel to load a
        different list of indices.
        """

        if self.use_potentials:
            return self.potential_item(batch_i)
        else:
            return self.random_item(batch_i)

    def potential_item(self, batch_i, debug_workers=False):

        t = [time.time()]

        # Initiate concatanation lists
        p_list = []
        f_list = []
        fxyz_list = []
        l_list = []
        i_list = []
        pi_list = []
        ci_list = []
        s_list = []
        R_list = []
        batch_n = 0
        failed_attempts = 0
        if self.config.weak_supervision:
            points_weak_inds = []
        #     p_list_weak = []
        #     f_list_weak = []
        #     fxyz_list_weak = []
        #     l_list_weak = []
        #     i_list_weak = []
        #     pi_list_weak = []
        #     ci_list_weak = []
        #     s_list_weak = []
        #     R_list_weak = []

        info = get_worker_info()
        if info is not None:
            wid = info.id
        else:
            wid = None

        while True:

            t += [time.time()]

            if debug_workers:
                message = ''
                for wi in range(info.num_workers):
                    if wi == wid:
                        message += ' {:}X{:} '.format(bcolors.FAIL, bcolors.ENDC)
                    elif self.worker_waiting[wi] == 0:
                        message += '   '
                    elif self.worker_waiting[wi] == 1:
                        message += ' | '
                    elif self.worker_waiting[wi] == 2:
                        message += ' o '
                print(message)
                self.worker_waiting[wid] = 0

            with self.worker_lock:

                if debug_workers:
                    message = ''
                    for wi in range(info.num_workers):
                        if wi == wid:
                            message += ' {:}v{:} '.format(bcolors.OKGREEN, bcolors.ENDC)
                        elif self.worker_waiting[wi] == 0:
                            message += '   '
                        elif self.worker_waiting[wi] == 1:
                            message += ' | '
                        elif self.worker_waiting[wi] == 2:
                            message += ' o '
                    print(message)
                    self.worker_waiting[wid] = 1

                # Get potential minimum
                cloud_ind = int(torch.argmin(self.min_potentials))
                point_ind = int(self.argmin_potentials[cloud_ind])

                # Get potential points from tree structure
                pot_points = np.array(self.pot_trees[cloud_ind].data, copy=False)

                # Center point of input region
                center_point = np.copy(pot_points[point_ind, :].reshape(1, -1))

                # Add a small noise to center point
                if self.set != 'ERF':
                    center_point += np.clip(np.random.normal(scale=self.config.in_radius / 10, size=center_point.shape),
                                            -self.config.in_radius / 2,
                                            self.config.in_radius / 2)

                # Indices of points in input region
                pot_inds, dists = self.pot_trees[cloud_ind].query_radius(center_point,
                                                                         r=self.config.in_radius,
                                                                         return_distance=True)

                d2s = np.square(dists[0])
                pot_inds = pot_inds[0]

                # Update potentials (Tukey weights)
                if self.set != 'ERF':
                    tukeys = np.square(1 - d2s / np.square(self.config.in_radius))
                    tukeys[d2s > np.square(self.config.in_radius)] = 0
                    self.potentials[cloud_ind][pot_inds] += tukeys
                    min_ind = torch.argmin(self.potentials[cloud_ind])
                    self.min_potentials[[cloud_ind]] = self.potentials[cloud_ind][min_ind]
                    self.argmin_potentials[[cloud_ind]] = min_ind

            t += [time.time()]

            # Get points from tree structure
            points = np.array(self.input_trees[cloud_ind].data, copy=False)


            # Indices of points in input region
            input_inds = self.input_trees[cloud_ind].query_radius(center_point,
                                                                  r=self.config.in_radius)[0]

            t += [time.time()]

            # Number collected
            n = input_inds.shape[0]

            # Safe check for empty spheres
            if n < 2:
                failed_attempts += 1
                if failed_attempts > 100 * self.config.batch_num:
                    raise ValueError('It seems this dataset only containes empty input spheres')
                t += [time.time()]
                t += [time.time()]
                continue

            # Collect labels and colors
            input_points = (points[input_inds] - center_point).astype(np.float32)
            input_colors = self.input_colors[cloud_ind][input_inds]
            if self.set in ['test', 'ERF']:
                input_labels = np.zeros(input_points.shape[0])
            else:
                input_labels = self.input_labels[cloud_ind][input_inds]
                input_labels = np.array([self.label_to_idx[l] for l in input_labels])

            if self.config.weak_supervision:
                all_weak_inds = self.weak_inds[cloud_ind]
                # calculate the intersection of the input_inds and all_weak_inds
                weak_inds = np.intersect1d(input_inds, all_weak_inds)
                
                # calculate the difference of the input_inds and weak_inds
                weak_mask_inds = np.setdiff1d(input_inds, weak_inds)
                
                weak_inds4input_points = np.in1d(input_inds, weak_mask_inds)
                
                points_weak_inds += [weak_inds]
                
                weak_label = self.weak_learning_label
                
                # change label of weak points to weak_label
                input_labels[weak_inds4input_points] = weak_label
            
            t += [time.time()]

            # Data augmentation
            input_points, scale, R = self.augmentation_transform(input_points)

            # Color augmentation
            if np.random.rand() > self.config.augment_color:
                input_colors *= 0

            # Get original height as additional feature
            input_features = np.hstack((input_colors, input_points[:, 2:] + center_point[:, 2:])).astype(np.float32)
            input_features_xyz = np.hstack((input_points, input_points[:, 2:] + center_point[:, 2:], input_colors)).astype(np.float32)

            t += [time.time()]

            # Stack batch
            p_list += [input_points]
            f_list += [input_features]
            fxyz_list += [input_features_xyz]
            l_list += [input_labels]
            pi_list += [input_inds]
            i_list += [point_ind]
            ci_list += [cloud_ind]
            s_list += [scale]
            R_list += [R]
            
            # if self.config.weak_supervision:
            #     all_weak_inds = self.weak_inds[cloud_ind]
            #     # calculate the intersection of the input_inds and all_weak_inds
            #     weak_inds = np.intersect1d(input_inds, all_weak_inds)
                
                # if len(weak_inds) > 0:
                #     # Collect weak labels and colors
                #     weak_points = (points[weak_inds] - center_point).astype(np.float32)
                #     weak_colors = self.input_colors[cloud_ind][weak_inds]
                #     if self.set in ['test', 'ERF']:
                #         weak_labels = np.zeros(weak_points.shape[0])
                #     else:
                #         weak_labels = self.input_labels[cloud_ind][weak_inds]
                #         weak_labels = np.array([self.label_to_idx[l] for l in weak_labels])

                    # Data augmentation from the same point
                    # weak_inds4input_points = np.in1d(input_inds, weak_inds)
                    # weak_points = input_points[weak_inds4input_points]
                    
                #     weak_points, weak_scale, weak_R = self.augmentation_transform(weak_points)
                    
                #     # Color augmentation
                #     if np.random.rand() > self.config.augment_color:
                #         weak_colors *= 0
                    
                #     # Get original height as additional feature
                #     weak_features = np.hstack((weak_colors, weak_points[:, 2:] + center_point[:, 2:])).astype(np.float32)
                #     weak_features_xyz = np.hstack((weak_points, weak_points[:, 2:] + center_point[:, 2:], weak_colors)).astype(np.float32)

                #     # Stack weak batch
                #     p_list_weak += [weak_points]
                #     f_list_weak += [weak_features]
                #     fxyz_list_weak += [weak_features_xyz]
                #     l_list_weak += [weak_labels]
                #     pi_list_weak += [weak_inds]
                #     i_list_weak += [point_ind]
                #     ci_list_weak += [cloud_ind]
                #     s_list_weak += [scale]
                #     R_list_weak += [R]
                # else:
                #     print('No weak supervision for this cloud')
                #     print('Cloud name: ', self.cloud_names[cloud_ind])
                #     print('Point index: ', point_ind)
                #     print('Center point: ', center_point)
                #     print('Pot points: ', pot_points)
                #     print('Pot inds: ', pot_inds)
                #     print('Distances: ', dists)
                #     print('Min potential: ', self.min_potentials[cloud_ind])
                #     print('Argmin potential: ', self.argmin_potentials[cloud_ind])
                #     print('Input inds: ', input_inds)
                #     print('Input points: ', input_points)
                #     print('Input labels: ', input_labels)
                #     if self.config.weak_supervision:
                #         print('all Weak inds: ', all_weak_inds)
                #         print('Weak inds: ', weak_inds)
                #     raise ValueError('No weak supervision for this cloud')
                

            # Update batch size
            batch_n += n

            # In case batch is full, stop
            if batch_n > int(self.batch_limit):
                break

            # Randomly drop some points (act as an augmentation process and a safety for GPU memory consumption)
            # if n > int(self.batch_limit):
            #    input_inds = np.random.choice(input_inds, size=int(self.batch_limit) - 1, replace=False)
            #    n = input_inds.shape[0]

        ###################
        # Concatenate batch
        ###################

        stacked_points = np.concatenate(p_list, axis=0)
        features = np.concatenate(f_list, axis=0)
        features_xyz = np.concatenate(fxyz_list, axis=0)
        labels = np.concatenate(l_list, axis=0)
        point_inds = np.array(i_list, dtype=np.int32)
        cloud_inds = np.array(ci_list, dtype=np.int32)
        input_inds = np.concatenate(pi_list, axis=0)
        stack_lengths = np.array([pp.shape[0] for pp in p_list], dtype=np.int32)
        scales = np.array(s_list, dtype=np.float32)
        rots = np.stack(R_list, axis=0)

        # Input features
        stacked_features = np.ones_like(stacked_points[:, :1], dtype=np.float32)
        if self.config.in_features_dim == 1:
            pass
        elif self.config.in_features_dim == 4:
            stacked_features = np.hstack((stacked_features, features[:, :3]))
        elif self.config.in_features_dim == 5:
            stacked_features = np.hstack((stacked_features, features))
        elif self.config.in_features_dim == 8:
            stacked_features = np.hstack((stacked_features, features_xyz))
        else:
            raise ValueError('Only accepted input dimensions are 1, 4 and 8 (without and with XYZ)')

        # if self.config.weak_supervision:
        #     stacked_weak_points = np.concatenate(p_list_weak, axis=0)
        #     weak_features = np.concatenate(f_list_weak, axis=0)
        #     weak_features_xyz = np.concatenate(fxyz_list_weak, axis=0)
        #     weak_labels = np.concatenate(l_list_weak, axis=0)
        #     weak_point_inds = np.array(i_list_weak, dtype=np.int32)
        #     weak_cloud_inds = np.array(ci_list_weak, dtype=np.int32)
        #     weak_input_inds = np.concatenate(pi_list_weak, axis=0)
        #     stacked_weak_lengths = np.array([pp.shape[0] for pp in p_list_weak], dtype=np.int32)
        #     weak_scales = np.array(s_list_weak, dtype=np.float32)
        #     weak_rots = np.stack(R_list_weak, axis=0)
            
        #     stacked_weak_features = np.ones_like(stacked_weak_points[:, :1], dtype=np.float32)
        #     if self.config.in_features_dim == 1:
        #         pass
        #     elif self.config.in_features_dim == 4:
        #         stacked_weak_features = np.hstack((stacked_weak_features, weak_features[:, :3]))
        #     elif self.config.in_features_dim == 5:
        #         stacked_weak_features = np.hstack((stacked_weak_features, weak_features))
        #     elif self.config.in_features_dim == 8:
        #         stacked_weak_features = np.hstack((stacked_weak_features, weak_features_xyz))
        #     else:
        #         raise ValueError('Only accepted input dimensions are 1, 4 and 8 (without and with XYZ)')

        #######################
        # Create network inputs
        #######################
        #
        #   Points, neighbors, pooling indices for each layers
        #

        t += [time.time()]

        # Get the whole input list
        input_list = self.segmentation_inputs(stacked_points,
                                              stacked_features,
                                              labels,
                                              stack_lengths)

        t += [time.time()]

        # Add scale and rotation for testing
        input_list += [scales, rots, cloud_inds, point_inds, input_inds]

        # if self.config.weak_supervision:
        #     input_weak_list = self.segmentation_inputs(stacked_weak_points,
        #                                                stacked_weak_features,
        #                                                weak_labels,
        #                                                stacked_weak_lengths)
        #     input_weak_list += [weak_scales, weak_rots, weak_cloud_inds, weak_point_inds, weak_input_inds]
        
        if debug_workers:
            message = ''
            for wi in range(info.num_workers):
                if wi == wid:
                    message += ' {:}0{:} '.format(bcolors.OKBLUE, bcolors.ENDC)
                elif self.worker_waiting[wi] == 0:
                    message += '   '
                elif self.worker_waiting[wi] == 1:
                    message += ' | '
                elif self.worker_waiting[wi] == 2:
                    message += ' o '
            print(message)
            self.worker_waiting[wid] = 2

        t += [time.time()]

        # Display timings
        debugT = False
        if debugT:
            print('\n************************\n')
            print('Timings:')
            ti = 0
            N = 5
            mess = 'Init ...... {:5.1f}ms /'
            loop_times = [1000 * (t[ti + N * i + 1] - t[ti + N * i]) for i in range(len(stack_lengths))]
            for dt in loop_times:
                mess += ' {:5.1f}'.format(dt)
            print(mess.format(np.sum(loop_times)))
            ti += 1
            mess = 'Pots ...... {:5.1f}ms /'
            loop_times = [1000 * (t[ti + N * i + 1] - t[ti + N * i]) for i in range(len(stack_lengths))]
            for dt in loop_times:
                mess += ' {:5.1f}'.format(dt)
            print(mess.format(np.sum(loop_times)))
            ti += 1
            mess = 'Sphere .... {:5.1f}ms /'
            loop_times = [1000 * (t[ti + N * i + 1] - t[ti + N * i]) for i in range(len(stack_lengths))]
            for dt in loop_times:
                mess += ' {:5.1f}'.format(dt)
            print(mess.format(np.sum(loop_times)))
            ti += 1
            mess = 'Collect ... {:5.1f}ms /'
            loop_times = [1000 * (t[ti + N * i + 1] - t[ti + N * i]) for i in range(len(stack_lengths))]
            for dt in loop_times:
                mess += ' {:5.1f}'.format(dt)
            print(mess.format(np.sum(loop_times)))
            ti += 1
            mess = 'Augment ... {:5.1f}ms /'
            loop_times = [1000 * (t[ti + N * i + 1] - t[ti + N * i]) for i in range(len(stack_lengths))]
            for dt in loop_times:
                mess += ' {:5.1f}'.format(dt)
            print(mess.format(np.sum(loop_times)))
            ti += N * (len(stack_lengths) - 1) + 1
            print('concat .... {:5.1f}ms'.format(1000 * (t[ti+1] - t[ti])))
            ti += 1
            print('input ..... {:5.1f}ms'.format(1000 * (t[ti+1] - t[ti])))
            ti += 1
            print('stack ..... {:5.1f}ms'.format(1000 * (t[ti+1] - t[ti])))
            ti += 1
            print('\n************************\n')
        if self.config.weak_supervision:
            input_list += [points_weak_inds]
        #     ret = {
        #         0: input_list,
        #         1: input_weak_list
        #     }
        #     return ret
        return input_list

    def random_item(self, batch_i):

        # Initiate concatanation lists
        p_list = []
        f_list = []
        l_list = []
        i_list = []
        pi_list = []
        ci_list = []
        s_list = []
        R_list = []
        batch_n = 0
        failed_attempts = 0

        while True:

            with self.worker_lock:

                # Get potential minimum
                cloud_ind = int(self.epoch_inds[0, self.epoch_i])
                point_ind = int(self.epoch_inds[1, self.epoch_i])

                # Update epoch indice
                self.epoch_i += 1
                if self.epoch_i >= int(self.epoch_inds.shape[1]):
                    self.epoch_i -= int(self.epoch_inds.shape[1])
                

            # Get points from tree structure
            points = np.array(self.input_trees[cloud_ind].data, copy=False)

            # Center point of input region
            center_point = np.copy(points[point_ind, :].reshape(1, -1))

            # Add a small noise to center point
            if self.set != 'ERF':
                center_point += np.clip(np.random.normal(scale=self.config.in_radius / 10, size=center_point.shape),
                                        -self.config.in_radius / 2,
                                        self.config.in_radius / 2)

            # Indices of points in input region
            input_inds = self.input_trees[cloud_ind].query_radius(center_point,
                                                                  r=self.config.in_radius)[0]

            # Number collected
            n = input_inds.shape[0]
            
            # Safe check for empty spheres
            if n < 2:
                failed_attempts += 1
                if failed_attempts > 100 * self.config.batch_num:
                    raise ValueError('It seems this dataset only containes empty input spheres')
                continue

            # Collect labels and colors
            input_points = (points[input_inds] - center_point).astype(np.float32)
            input_colors = self.input_colors[cloud_ind][input_inds]
            if self.set in ['test', 'ERF']:
                input_labels = np.zeros(input_points.shape[0])
            else:
                input_labels = self.input_labels[cloud_ind][input_inds]
                input_labels = np.array([self.label_to_idx[l] for l in input_labels])

            # Data augmentation
            input_points, scale, R = self.augmentation_transform(input_points)

            # Color augmentation
            if np.random.rand() > self.config.augment_color:
                input_colors *= 0

            # Get original height as additional feature
            input_features = np.hstack((input_colors, input_points[:, 2:] + center_point[:, 2:])).astype(np.float32)

            # Stack batch
            p_list += [input_points]
            f_list += [input_features]
            l_list += [input_labels]
            pi_list += [input_inds]
            i_list += [point_ind]
            ci_list += [cloud_ind]
            s_list += [scale]
            R_list += [R]

            # Update batch size
            batch_n += n

            # In case batch is full, stop
            if batch_n > int(self.batch_limit):
                break

            # Randomly drop some points (act as an augmentation process and a safety for GPU memory consumption)
            # if n > int(self.batch_limit):
            #    input_inds = np.random.choice(input_inds, size=int(self.batch_limit) - 1, replace=False)
            #    n = input_inds.shape[0]

        ###################
        # Concatenate batch
        ###################

        stacked_points = np.concatenate(p_list, axis=0)
        features = np.concatenate(f_list, axis=0)
        labels = np.concatenate(l_list, axis=0)
        point_inds = np.array(i_list, dtype=np.int32)
        cloud_inds = np.array(ci_list, dtype=np.int32)
        input_inds = np.concatenate(pi_list, axis=0)
        stack_lengths = np.array([pp.shape[0] for pp in p_list], dtype=np.int32)
        scales = np.array(s_list, dtype=np.float32)
        rots = np.stack(R_list, axis=0)

        # Input features
        stacked_features = np.ones_like(stacked_points[:, :1], dtype=np.float32)
        if self.config.in_features_dim == 1:
            pass
        elif self.config.in_features_dim == 4:
            stacked_features = np.hstack((stacked_features, features[:, :3]))
        elif self.config.in_features_dim == 5:
            stacked_features = np.hstack((stacked_features, features))
        else:
            raise ValueError('Only accepted input dimensions are 1, 4 and 7 (without and with XYZ)')

        #######################
        # Create network inputs
        #######################
        #
        #   Points, neighbors, pooling indices for each layers
        #

        # Get the whole input list
        input_list = self.segmentation_inputs(stacked_points,
                                              stacked_features,
                                              labels,
                                              stack_lengths)

        # Add scale and rotation for testing
        input_list += [scales, rots, cloud_inds, point_inds, input_inds]

        return input_list

    def prepare_ISPRS_ply(self):

        print('\nPreparing ply files')
        t0 = time.time()

        # Folder for the ply files
        ply_path = join(self.path, self.train_path)
        if not exists(ply_path):
            makedirs(ply_path)

        for cloud_name in self.cloud_names:

            # Pass if the cloud has already been computed
            cloud_file = join(ply_path, cloud_name + '.ply')
            if exists(cloud_file):
                continue
            else:
                # Initiate containers
                cloud_points = np.empty((0, 3), dtype=np.float32)
                cloud_features = np.empty((0, 3), dtype=np.uint8)
                cloud_classes = np.empty((0, 1), dtype=np.int32)

                pts_file = join(self.path, cloud_name + '.pts')
                
                if not exists(pts_file):
                    raise ValueError('File not found: ' + pts_file)
                # Read pts file
                points, features, labels = read_pts_file(pts_file)
                
                # Convert to numpy
                points = np.array(points, dtype=np.float32)
                features = np.array(features,dtype=np.uint8)
                labels = np.array(labels, dtype=np.int32)
                
                # Separate the x, y, z coordinates
                cloud_x = points[:, 0]
                cloud_y = points[:, 1]
                cloud_z = points[:, 2]

                # Shift the coordinates to make the minimum 0
                cloud_x = cloud_x - cloud_x.min()
                cloud_y = cloud_y - cloud_y.min()
                cloud_z = cloud_z - cloud_z.min()

                # Combine the shifted coordinates back into cloud_points
                points = np.stack((cloud_x, cloud_y, cloud_z), axis=-1)
                
                # Stack all data
                cloud_points = np.vstack((cloud_points, points.astype(np.float32)))
                cloud_features = np.vstack((cloud_features, features.astype(np.uint8)))
                cloud_classes = np.vstack((cloud_classes, labels.reshape(-1, 1).astype(np.int32)))

                # Save as ply
                write_ply(cloud_file,
                        (cloud_points, cloud_features, cloud_classes),
                        ['x', 'y', 'z', 'f1', 'f2', 'f3', 'class'])

        print('Done in {:.1f}s'.format(time.time() - t0))
        return

    def load_subsampled_clouds(self):

        # Parameter
        dl = self.config.first_subsampling_dl

        # Create path for files
        tree_path = join(self.path, 'input_{:.3f}'.format(dl))
        if not exists(tree_path):
            makedirs(tree_path)

        ##############
        # Load KDTrees
        ##############

        for i, file_path in enumerate(self.files):

            # Restart timer
            t0 = time.time()

            # Get cloud name
            cloud_name = self.cloud_names[i]

            # Name of the input files
            KDTree_file = join(tree_path, '{:s}_KDTree.pkl'.format(cloud_name))
            sub_ply_file = join(tree_path, '{:s}.ply'.format(cloud_name))

            # Check if inputs have already been computed
            if exists(KDTree_file):
                print('\nFound KDTree for cloud {:s}, subsampled at {:.3f}'.format(cloud_name, dl))

                # read ply with data
                data = read_ply(sub_ply_file)
                sub_features = np.vstack((data['f1'], data['f2'], data['f3'])).T
                sub_labels = data['class']
                
                sub_points = np.vstack((data['x'], data['y'], data['z'])).T

                # Read pkl with search tree
                with open(KDTree_file, 'rb') as f:
                    search_tree = pickle.load(f)

            else:
                print('\nPreparing KDTree for cloud {:s}, subsampled at {:.3f}'.format(cloud_name, dl))

                # Read ply file
                data = read_ply(file_path)
                points = np.vstack((data['x'], data['y'], data['z'])).T
                features = np.vstack((data['f1'], data['f2'], data['f3'])).T
                labels = data['class']

                # Subsample cloud
                sub_points, sub_features, sub_labels = grid_subsampling(points,
                                                        features= features,
                                                        labels=labels,
                                                        sampleDl=dl)

                # Squeeze label
                sub_labels = np.squeeze(sub_labels)

                # Get chosen neighborhoods
                search_tree = KDTree(sub_points, leaf_size=10)
                #search_tree = nnfln.KDTree(n_neighbors=1, metric='L2', leaf_size=10)
                #search_tree.fit(sub_points)

                # Save KDTree
                with open(KDTree_file, 'wb') as f:
                    pickle.dump(search_tree, f)

                # Save ply
                write_ply(sub_ply_file,
                          [sub_points, sub_features, sub_labels],
                          ['x', 'y', 'z', 'f1', 'f2', 'f3', 'class'])
           
            if self.config.weak_supervision:
                select_method = self.config.weak_select_method
                percentage = self.config.weak_supervision_perc
                ws_in_radius = self.config.weak_supervision_in_radius
                weak_supervision_inds_file_name = join(tree_path, '{:s}_ws_inds{:.5f}_radius{:.5f}_{:s}'.format(cloud_name, percentage, ws_in_radius, select_method))
                weak_supervision_inds_file = weak_supervision_inds_file_name + '.npy'
                # Create weak supervision labels index
                # Uniform distribution of labels
                if select_method == 'ball_random':
                    selected_inds = self.select_weak_inds_ball_random(weak_supervision_inds_file, sub_points, sub_labels, search_tree, ws_in_radius, percentage)
                elif select_method == 'avg_random':
                    selected_inds = self.select_weak_inds_avg_random(weak_supervision_inds_file, search_tree, sub_labels, percentage)
                elif select_method == 'avg_distance':
                    selected_inds = self.select_weak_inds_avg_distance(weak_supervision_inds_file, sub_points, search_tree, sub_labels, percentage)
                self.weak_inds += [selected_inds]
                
            
            # Fill data containers
            self.input_trees += [search_tree]
            self.input_colors += [sub_features]
            self.input_labels += [sub_labels]

            size = sub_labels.shape[0] * 4 * 7
            print('{:.1f} MB loaded in {:.1f}s'.format(size * 1e-6, time.time() - t0))

        ############################
        # Coarse potential locations
        ############################

        # Only necessary for validation and test sets
        if self.use_potentials:
            print('\nPreparing potentials')

            # Restart timer
            t0 = time.time()

            pot_dl = self.config.in_radius / 10
            cloud_ind = 0

            for i, file_path in enumerate(self.files):

                # Get cloud name
                cloud_name = self.cloud_names[i]

                # Name of the input files
                coarse_KDTree_file = join(tree_path, '{:s}_coarse_KDTree.pkl'.format(cloud_name))

                # Check if inputs have already been computed
                if exists(coarse_KDTree_file):
                    # Read pkl with search tree
                    with open(coarse_KDTree_file, 'rb') as f:
                        search_tree = pickle.load(f)

                else:
                    # Subsample cloud
                    sub_points = np.array(self.input_trees[cloud_ind].data, copy=False)
                    coarse_points = grid_subsampling(sub_points.astype(np.float32), sampleDl=pot_dl)

                    # Get chosen neighborhoods
                    search_tree = KDTree(coarse_points, leaf_size=10)

                    # Save KDTree
                    with open(coarse_KDTree_file, 'wb') as f:
                        pickle.dump(search_tree, f)

                # Fill data containers
                self.pot_trees += [search_tree]
                cloud_ind += 1

            print('Done in {:.1f}s'.format(time.time() - t0))

        ######################
        # Reprojection indices
        ######################

        # Get number of clouds
        self.num_clouds = len(self.input_trees)

        # Only necessary for validation and test sets
        if self.set in ['validation', 'test']:

            print('\nPreparing reprojection indices for testing')

            # Get validation/test reprojection indices
            for i, file_path in enumerate(self.files):

                # Restart timer
                t0 = time.time()

                # Get info on this cloud
                cloud_name = self.cloud_names[i]

                # File name for saving
                proj_file = join(tree_path, '{:s}_proj.pkl'.format(cloud_name))

                # Try to load previous indices
                if exists(proj_file):
                    with open(proj_file, 'rb') as f:
                        proj_inds, labels = pickle.load(f)
                else:
                    data = read_ply(file_path)
                    points = np.vstack((data['x'], data['y'], data['z'])).T
                    labels = data['class']

                    # Compute projection inds
                    idxs = self.input_trees[i].query(points, return_distance=False)
                    #dists, idxs = self.input_trees[i_cloud].kneighbors(points)
                    proj_inds = np.squeeze(idxs).astype(np.int32)

                    # Save
                    with open(proj_file, 'wb') as f:
                        pickle.dump([proj_inds, labels], f)

                self.test_proj += [proj_inds]
                self.validation_labels += [labels]
                print('{:s} done in {:.1f}s'.format(cloud_name, time.time() - t0))

        print()
        return

    def load_evaluation_points(self, file_path):
        """
        Load points (from test or validation split) on which the metrics should be evaluated
        """

        # Get original points
        data = read_ply(file_path)
        return np.vstack((data['x'], data['y'], data['z'])).T
    
    def select_weak_inds_ball_random(self, weak_supervision_inds_file, sub_points, sub_labels, search_tree, ws_in_radius, percentage):
        if exists(weak_supervision_inds_file):
            with open(weak_supervision_inds_file, 'rb') as f:
                selected_inds = np.load(f)
            selected_labels = sub_labels[selected_inds]
            selected_counts = np.zeros(self.config.num_classes, dtype=np.int32)
            for l in selected_labels:
                selected_counts[l] += 1
            cloud_counts = np.zeros(self.config.num_classes, dtype=np.int32)
            for l in sub_labels:
                cloud_counts[l] += 1
            print('selected_counts', selected_counts)
            print('selected_counts / cloud_counts', selected_counts / cloud_counts)
            print('all selected_counts:', selected_counts.sum(), 'all cloud_counts:', cloud_counts.sum())
            print('selected_counts / cloud_counts:', selected_counts.sum() / cloud_counts.sum())
        else:
            points_sum = search_tree.data.shape[0]
            print('all points', points_sum)
            selected_inds = np.ndarray(shape=(0,), dtype=np.int32)
            selected_counts = np.zeros(self.config.num_classes, dtype=np.int32)
            cloud_counts = np.ones(self.config.num_classes, dtype=np.int32)
            
            # check if the selected points are covered all classes(every class is bigger than 0)
            while np.any(selected_counts == 0):
                print('not enough selected points, reselecting')
                selected_inds = np.ndarray(shape=(0,), dtype=np.int32)
                while selected_inds.shape[0] < int(points_sum * percentage):
                    # Get points in the ball
                    center_point_ind = np.random.randint(points_sum)
                    inds = search_tree.query_radius(sub_points[center_point_ind].reshape(1, -1),
                                                    r=ws_in_radius)[0]
                    selected_inds = np.concatenate((selected_inds, inds))
                    selected_inds = np.unique(selected_inds)
                print('selected_inds.shape', selected_inds.shape)
                    
                # count the number of selected points by class
                selected_labels = sub_labels[selected_inds]
                selected_counts = np.zeros(self.config.num_classes, dtype=np.int32)
                for l in selected_labels:
                    selected_counts[l] += 1
                print('selected_counts', selected_counts)
                
                # count the number of points in each class in the whole cloud
                cloud_labels = sub_labels
                cloud_counts = np.zeros(self.config.num_classes, dtype=np.int32)
                for l in cloud_labels:
                    cloud_counts[l] += 1
                print('cloud_counts', cloud_counts)
                
                print('selected_counts / cloud_counts')
                print(selected_counts / cloud_counts)
                print('all selected_counts:', selected_counts.sum(), 'all cloud_counts:', cloud_counts.sum())
                print('selected_counts / cloud_counts:', selected_counts.sum() / cloud_counts.sum())
            
            # Save weak supervision indices
            with open(weak_supervision_inds_file, 'wb') as f:
                np.save(f, selected_inds)
        return selected_inds
    
    def select_weak_inds_avg_random(self, weak_supervision_inds_file, search_tree, sub_labels, percentage):
        if exists(weak_supervision_inds_file):
            with open(weak_supervision_inds_file, 'rb') as f:
                selected_inds = np.load(f)
            selected_labels = sub_labels[selected_inds]
            selected_counts = np.zeros(self.config.num_classes, dtype=np.int32)
            for l in selected_labels:
                selected_counts[l] += 1
            cloud_counts = np.zeros(self.config.num_classes, dtype=np.int32)
            for l in sub_labels:
                cloud_counts[l] += 1
            print('selected_counts', selected_counts)
            print('selected_counts / cloud_counts', selected_counts / cloud_counts)
            print('all selected_counts:', selected_counts.sum(), 'all cloud_counts:', cloud_counts.sum())
            print('selected_counts / cloud_counts:', selected_counts.sum() / cloud_counts.sum())
        else:
            points_sum = search_tree.data.shape[0]
            print('all points', points_sum)
            selected_inds = np.ndarray(shape=(0,), dtype=np.int32)
            selected_counts = np.zeros(self.config.num_classes, dtype=np.int32)
            cloud_counts = np.ones(self.config.num_classes, dtype=np.int32)

            each_class_num = int(points_sum * percentage)//self.config.num_classes
            print('each_class_num', each_class_num)
            each_class_inds = []
            for i in range(self.config.num_classes):
                each_class_inds.append(np.where(sub_labels == i)[0])
            for i in range(self.config.num_classes):
                inds = each_class_inds[i]
                if len(inds) < each_class_num:
                    selected_inds = np.concatenate((selected_inds, inds))
                else:
                    class_selected_inds = np.random.choice(inds, size=each_class_num, replace=False)
                    selected_inds = np.concatenate((selected_inds, class_selected_inds))
            
            # count the number of selected points by class
            for l in sub_labels[selected_inds]:
                selected_counts[l] += 1
            for l in sub_labels:
                cloud_counts[l] += 1
            print('selected_counts', selected_counts)
            print('selected_counts / cloud_counts', selected_counts / cloud_counts)
            print('all selected_counts:', selected_counts.sum(), 'all cloud_counts:', cloud_counts.sum())
            print('selected_counts / cloud_counts:', selected_counts.sum() / cloud_counts.sum())
            
            # Save weak supervision indices
            with open(weak_supervision_inds_file, 'wb') as f:
                np.save(f, selected_inds)
        return selected_inds
    
    def select_weak_inds_avg_distance(self, weak_supervision_inds_file, sub_points, search_tree, sub_labels, percentage):
        if exists(weak_supervision_inds_file):
            with open(weak_supervision_inds_file, 'rb') as f:
                selected_inds = np.load(f)
            selected_labels = sub_labels[selected_inds]
            selected_counts = np.zeros(self.config.num_classes, dtype=np.int32)
            for l in selected_labels:
                selected_counts[l] += 1
            cloud_counts = np.zeros(self.config.num_classes, dtype=np.int32)
            for l in sub_labels:
                cloud_counts[l] += 1
            print('selected_counts', selected_counts)
            print('selected_counts / cloud_counts', selected_counts / cloud_counts)
            print('all selected_counts:', selected_counts.sum(), 'all cloud_counts:', cloud_counts.sum())
            print('selected_counts / cloud_counts:', selected_counts.sum() / cloud_counts.sum())
        else:
            points_sum = search_tree.data.shape[0]
            print('all points', points_sum)
            selected_inds = np.ndarray(shape=(0,), dtype=np.int32)
            selected_counts = np.zeros(self.config.num_classes, dtype=np.int32)
            cloud_counts = np.ones(self.config.num_classes, dtype=np.int32)

            each_class_num = int(points_sum * percentage)//self.config.num_classes
            print('each_class_num', each_class_num)
            each_class_inds = []
            for i in range(self.config.num_classes):
                each_class_inds.append(np.where(sub_labels == i)[0])
            longest_dist = 0
            # find the longest distance between all points in search_tree
            dist = 1
            center_point_ind = np.random.choice(search_tree.data.shape[0])
            inds_fomer = search_tree.query_radius(sub_points[center_point_ind].reshape(1, -1), r=dist)[0]
            dist += 5
            while True:
                inds = search_tree.query_radius(sub_points[center_point_ind].reshape(1, -1), r=dist)[0]
                if len(inds) == len(inds_fomer):
                    break
                else:
                    inds_fomer = inds
                dist += 5
            longest_dist = dist*1.5
            print('longest_dist', longest_dist)
            avg_dist = longest_dist / each_class_num
            bias = avg_dist * 0.05
            all_neighbor_inds = np.ndarray(shape=(0,), dtype=np.int32)
            while i in range(self.config.num_classes):
                inds = each_class_inds[i]
                class_selected_inds = np.ndarray(shape=(0,), dtype=np.int32)
                if len(inds) < each_class_num:
                    selected_inds = np.concatenate((selected_inds, inds))
                else:
                    center_point_ind = np.random.choice(inds)
                    class_selected_inds = np.array([center_point_ind])
                    while len(class_selected_inds) < each_class_num:
                        inds_in_ball = search_tree.query_radius(sub_points[center_point_ind].reshape(1, -1), r=avg_dist)[0]
                        inds_inin_ball = search_tree.query_radius(sub_points[center_point_ind].reshape(1, -1), r=avg_dist-bias)[0]
                        inds_shell_1 = np.setdiff1d(inds_in_ball, inds_inin_ball)
                        if len(inds_shell_1) == 0:
                            j = np.random.choice(inds)
                        inds_shell = np.setdiff1d(inds_shell_1, all_neighbor_inds)
                        if len(inds_shell) == 0:
                            j = np.random.choice(inds_shell_1)
                            center_point_ind = j
                            continue
                        j = np.random.choice(inds_shell)
                        center_point_ind = j
                        class_selected_inds = np.concatenate((class_selected_inds, [center_point_ind]))
                        all_neighbor_inds = np.concatenate((all_neighbor_inds, inds_inin_ball))
                    selected_inds = np.concatenate((selected_inds, class_selected_inds))
                    # unique
                    all_neighbor_inds = np.unique(all_neighbor_inds)
            # count the number of selected points by class
            for l in sub_labels[selected_inds]:
                selected_counts[l] += 1
            for l in sub_labels:
                cloud_counts[l] += 1
            print('selected_counts', selected_counts)
            print('selected_counts / cloud_counts', selected_counts / cloud_counts)
            print('all selected_counts:', selected_counts.sum(), 'all cloud_counts:', cloud_counts.sum())
            print('selected_counts / cloud_counts:', selected_counts.sum() / cloud_counts.sum())
            
            # Save weak supervision indices
            with open(weak_supervision_inds_file, 'wb') as f:
                np.save(f, selected_inds)
        return selected_inds

# ----------------------------------------------------------------------------------------------------------------------
#
#           Utility classes definition
#       \********************************/


class ISPRSSampler(Sampler):
    """Sampler for ISPRS"""

    def __init__(self, dataset: ISPRSDataset):
        Sampler.__init__(self, dataset)

        # Dataset used by the sampler (no copy is made in memory)
        self.dataset = dataset

        # Number of step per epoch
        if dataset.set == 'training':
            self.N = dataset.config.epoch_steps
        else:
            self.N = dataset.config.validation_size

        return

    def __iter__(self):
        """
        Yield next batch indices here. In this dataset, this is a dummy sampler that yield the index of batch element
        (input sphere) in epoch instead of the list of point indices
        """

        if not self.dataset.use_potentials:

            # Initiate current epoch ind
            self.dataset.epoch_i *= 0
            self.dataset.epoch_inds *= 0

            # Initiate container for indices
            all_epoch_inds = np.zeros((2, 0), dtype=np.int64)

            # Number of sphere centers taken per class in each cloud
            num_centers = self.N * self.dataset.config.batch_num
            random_pick_n = int(np.ceil(num_centers / self.dataset.config.num_classes))

            # Choose random points of each class for each cloud
            epoch_indices = np.zeros((2, 0), dtype=np.int64)
            for label_ind, label in enumerate(self.dataset.label_values):
                if label not in self.dataset.ignored_labels:

                    # Gather indices of the points with this label in all the input clouds 
                    all_label_indices = []
                    for cloud_ind, cloud_labels in enumerate(self.dataset.input_labels):
                        label_indices = np.where(np.equal(cloud_labels, label))[0]
                        all_label_indices.append(np.vstack((np.full(label_indices.shape, cloud_ind, dtype=np.int64), label_indices)))

                    # Stack them: [2, N1+N2+...]
                    all_label_indices = np.hstack(all_label_indices)

                    # Select a a random number amongst them
                    N_inds = all_label_indices.shape[1]
                    if N_inds < random_pick_n:
                        chosen_label_inds = np.zeros((2, 0), dtype=np.int64)
                        while chosen_label_inds.shape[1] < random_pick_n:
                            chosen_label_inds = np.hstack((chosen_label_inds, all_label_indices[:, np.random.permutation(N_inds)]))
                        warnings.warn('When choosing random epoch indices (use_potentials=False), \
                                       class {:d}: {:s} only had {:d} available points, while we \
                                       needed {:d}. Repeating indices in the same epoch'.format(label,
                                                                                                self.dataset.label_names[label_ind],
                                                                                                N_inds,
                                                                                                random_pick_n))

                    elif N_inds < 50 * random_pick_n:
                        rand_inds = np.random.choice(N_inds, size=random_pick_n, replace=False)
                        chosen_label_inds = all_label_indices[:, rand_inds]

                    else:
                        chosen_label_inds = np.zeros((2, 0), dtype=np.int64)
                        while chosen_label_inds.shape[1] < random_pick_n:
                            rand_inds = np.unique(np.random.choice(N_inds, size=2*random_pick_n, replace=True))
                            chosen_label_inds = np.hstack((chosen_label_inds, all_label_indices[:, rand_inds]))
                        chosen_label_inds = chosen_label_inds[:, :random_pick_n]

                    # Stack for each label
                    all_epoch_inds = np.hstack((all_epoch_inds, chosen_label_inds))

            # Random permutation of the indices
            random_order = np.random.permutation(all_epoch_inds.shape[1])[:num_centers]
            all_epoch_inds = all_epoch_inds[:, random_order].astype(np.int64)

            # Update epoch inds
            self.dataset.epoch_inds += torch.from_numpy(all_epoch_inds)

        # Generator loop
        for i in range(self.N):
            yield i

    def __len__(self):
        """
        The number of yielded samples is variable
        """
        return self.N

    def fast_calib(self):
        """
        This method calibrates the batch sizes while ensuring the potentials are well initialized. Indeed on a dataset
        like Semantic3D, before potential have been updated over the dataset, there are cahnces that all the dense area
        are picked in the begining and in the end, we will have very large batch of small point clouds
        :return:
        """

        # Estimated average batch size and target value
        estim_b = 0
        target_b = self.dataset.config.batch_num

        # Calibration parameters
        low_pass_T = 10
        Kp = 100.0
        finer = False
        breaking = False

        # Convergence parameters
        smooth_errors = []
        converge_threshold = 0.1

        t = [time.time()]
        last_display = time.time()
        mean_dt = np.zeros(2)

        for epoch in range(10):
            for i, test in enumerate(self):

                # New time
                t = t[-1:]
                t += [time.time()]

                # batch length
                b = len(test)

                # Update estim_b (low pass filter)
                estim_b += (b - estim_b) / low_pass_T

                # Estimate error (noisy)
                error = target_b - b

                # Save smooth errors for convergene check
                smooth_errors.append(target_b - estim_b)
                if len(smooth_errors) > 10:
                    smooth_errors = smooth_errors[1:]

                # Update batch limit with P controller
                self.dataset.batch_limit += Kp * error

                # finer low pass filter when closing in
                if not finer and np.abs(estim_b - target_b) < 1:
                    low_pass_T = 100
                    finer = True

                # Convergence
                if finer and np.max(np.abs(smooth_errors)) < converge_threshold:
                    breaking = True
                    break

                # Average timing
                t += [time.time()]
                mean_dt = 0.9 * mean_dt + 0.1 * (np.array(t[1:]) - np.array(t[:-1]))

                # Console display (only one per second)
                if (t[-1] - last_display) > 1.0:
                    last_display = t[-1]
                    message = 'Step {:5d}  estim_b ={:5.2f} batch_limit ={:7d},  //  {:.1f}ms {:.1f}ms'
                    print(message.format(i,
                                         estim_b,
                                         int(self.dataset.batch_limit),
                                         1000 * mean_dt[0],
                                         1000 * mean_dt[1]))

            if breaking:
                break

    def calibration(self, dataloader, untouched_ratio=0.9, verbose=False, force_redo=False):
        """
        Method performing batch and neighbors calibration.
            Batch calibration: Set "batch_limit" (the maximum number of points allowed in every batch) so that the
                               average batch size (number of stacked pointclouds) is the one asked.
        Neighbors calibration: Set the "neighborhood_limits" (the maximum number of neighbors allowed in convolutions)
                               so that 90% of the neighborhoods remain untouched. There is a limit for each layer.
        """

        ##############################
        # Previously saved calibration
        ##############################

        print('\nStarting Calibration (use verbose=True for more details)')
        t0 = time.time()

        redo = force_redo

        # Batch limit
        # ***********

        # Load batch_limit dictionary
        batch_lim_file = join(self.dataset.path, 'batch_limits.pkl')
        if exists(batch_lim_file):
            with open(batch_lim_file, 'rb') as file:
                batch_lim_dict = pickle.load(file)
        else:
            batch_lim_dict = {}

        # Check if the batch limit associated with current parameters exists
        if self.dataset.use_potentials:
            sampler_method = 'potentials'
        else:
            sampler_method = 'random'
        key = '{:s}_{:.3f}_{:.3f}_{:d}'.format(sampler_method,
                                               self.dataset.config.in_radius,
                                               self.dataset.config.first_subsampling_dl,
                                               self.dataset.config.batch_num)
        if not redo and key in batch_lim_dict:
            self.dataset.batch_limit[0] = batch_lim_dict[key]
        else:
            redo = True

        if verbose:
            print('\nPrevious calibration found:')
            print('Check batch limit dictionary')
            if key in batch_lim_dict:
                color = bcolors.OKGREEN
                v = str(int(batch_lim_dict[key]))
            else:
                color = bcolors.FAIL
                v = '?'
            print('{:}\"{:s}\": {:s}{:}'.format(color, key, v, bcolors.ENDC))

        # Neighbors limit
        # ***************

        # Load neighb_limits dictionary
        neighb_lim_file = join(self.dataset.path, 'neighbors_limits.pkl')
        if exists(neighb_lim_file):
            with open(neighb_lim_file, 'rb') as file:
                neighb_lim_dict = pickle.load(file)
        else:
            neighb_lim_dict = {}

        # Check if the limit associated with current parameters exists (for each layer)
        neighb_limits = []
        for layer_ind in range(self.dataset.config.num_layers):

            dl = self.dataset.config.first_subsampling_dl * (2**layer_ind)
            if self.dataset.config.deform_layers[layer_ind]:
                r = dl * self.dataset.config.deform_radius
            else:
                r = dl * self.dataset.config.conv_radius

            key = '{:.3f}_{:.3f}'.format(dl, r)
            if key in neighb_lim_dict:
                neighb_limits += [neighb_lim_dict[key]]

        if not redo and len(neighb_limits) == self.dataset.config.num_layers:
            self.dataset.neighborhood_limits = neighb_limits
        else:
            redo = True

        if verbose:
            print('Check neighbors limit dictionary')
            for layer_ind in range(self.dataset.config.num_layers):
                dl = self.dataset.config.first_subsampling_dl * (2**layer_ind)
                if self.dataset.config.deform_layers[layer_ind]:
                    r = dl * self.dataset.config.deform_radius
                else:
                    r = dl * self.dataset.config.conv_radius
                key = '{:.3f}_{:.3f}'.format(dl, r)

                if key in neighb_lim_dict:
                    color = bcolors.OKGREEN
                    v = str(neighb_lim_dict[key])
                else:
                    color = bcolors.FAIL
                    v = '?'
                print('{:}\"{:s}\": {:s}{:}'.format(color, key, v, bcolors.ENDC))

        if redo:

            ############################
            # Neighbors calib parameters
            ############################

            # From config parameter, compute higher bound of neighbors number in a neighborhood
            hist_n = int(np.ceil(4 / 3 * np.pi * (self.dataset.config.deform_radius + 1) ** 3))

            # Histogram of neighborhood sizes
            neighb_hists = np.zeros((self.dataset.config.num_layers, hist_n), dtype=np.int32)

            ########################
            # Batch calib parameters
            ########################

            # Estimated average batch size and target value
            estim_b = 0
            target_b = self.dataset.config.batch_num
            
            # Expected batch size order of magnitude
            expected_N = 100000

            # Calibration parameters. Higher means faster but can also become unstable
            # Reduce Kp and Kd if your GP Uis small as the total number of points per batch will be smaller 
            low_pass_T = 100
            Kp = expected_N / 200
            Ki = 0.001 * Kp
            Kd = 5 * Kp
            finer = False
            stabilized = False

            # Convergence parameters
            smooth_errors = []
            converge_threshold = 0.1

            # Loop parameters
            last_display = time.time()
            i = 0
            breaking = False
            error_I = 0
            error_D = 0
            last_error = 0

            debug_in = []
            debug_out = []
            debug_b = []
            debug_estim_b = []

            #####################
            # Perform calibration
            #####################

            # number of batch per epoch 
            sample_batches = 999
            for epoch in range((sample_batches // self.N) + 1):
                for batch_i, batch in enumerate(dataloader):

                    # Update neighborhood histogram
                    counts = [np.sum(neighb_mat.numpy() < neighb_mat.shape[0], axis=1) for neighb_mat in batch.neighbors]
                    hists = [np.bincount(c, minlength=hist_n)[:hist_n] for c in counts]
                    neighb_hists += np.vstack(hists)

                    # batch length
                    b = len(batch.cloud_inds)

                    # Update estim_b (low pass filter)
                    estim_b += (b - estim_b) / low_pass_T

                    # Estimate error (noisy)
                    error = target_b - b
                    error_I += error
                    error_D = error - last_error
                    last_error = error


                    # Save smooth errors for convergene check
                    smooth_errors.append(target_b - estim_b)
                    if len(smooth_errors) > 30:
                        smooth_errors = smooth_errors[1:]

                    # Update batch limit with P controller
                    self.dataset.batch_limit += Kp * error + Ki * error_I + Kd * error_D

                    # Unstability detection
                    if not stabilized and self.dataset.batch_limit < 0:
                        Kp *= 0.1
                        Ki *= 0.1
                        Kd *= 0.1
                        stabilized = True

                    # finer low pass filter when closing in
                    if not finer and np.abs(estim_b - target_b) < 1:
                        low_pass_T = 100
                        finer = True

                    # Convergence
                    if finer and np.max(np.abs(smooth_errors)) < converge_threshold:
                        breaking = True
                        break

                    i += 1
                    t = time.time()

                    # Console display (only one per second)
                    if verbose and (t - last_display) > 1.0:
                        last_display = t
                        message = 'Step {:5d}  estim_b ={:5.2f} batch_limit ={:7d}'
                        print(message.format(i,
                                             estim_b,
                                             int(self.dataset.batch_limit)))

                    # Debug plots
                    debug_in.append(int(batch.points[0].shape[0]))
                    debug_out.append(int(self.dataset.batch_limit))
                    debug_b.append(b)
                    debug_estim_b.append(estim_b)

                if breaking:
                    break

            # Plot in case we did not reach convergence
            if not breaking:
                import matplotlib.pyplot as plt

                print("ERROR: It seems that the calibration have not reached convergence. Here are some plot to understand why:")
                print("If you notice unstability, reduce the expected_N value")
                print("If convergece is too slow, increase the expected_N value")

                plt.figure()
                plt.plot(debug_in)
                plt.plot(debug_out)

                plt.figure()
                plt.plot(debug_b)
                plt.plot(debug_estim_b)

                plt.show()

                a = 1/0


            # Use collected neighbor histogram to get neighbors limit
            cumsum = np.cumsum(neighb_hists.T, axis=0)
            percentiles = np.sum(cumsum < (untouched_ratio * cumsum[hist_n - 1, :]), axis=0)
            self.dataset.neighborhood_limits = percentiles


            if verbose:

                # Crop histogram
                while np.sum(neighb_hists[:, -1]) == 0:
                    neighb_hists = neighb_hists[:, :-1]
                hist_n = neighb_hists.shape[1]

                print('\n**************************************************\n')
                line0 = 'neighbors_num '
                for layer in range(neighb_hists.shape[0]):
                    line0 += '|  layer {:2d}  '.format(layer)
                print(line0)
                for neighb_size in range(hist_n):
                    line0 = '     {:4d}     '.format(neighb_size)
                    for layer in range(neighb_hists.shape[0]):
                        if neighb_size > percentiles[layer]:
                            color = bcolors.FAIL
                        else:
                            color = bcolors.OKGREEN
                        line0 += '|{:}{:10d}{:}  '.format(color,
                                                         neighb_hists[layer, neighb_size],
                                                         bcolors.ENDC)

                    print(line0)

                print('\n**************************************************\n')
                print('\nchosen neighbors limits: ', percentiles)
                print()

            # Save batch_limit dictionary
            if self.dataset.use_potentials:
                sampler_method = 'potentials'
            else:
                sampler_method = 'random'
            key = '{:s}_{:.3f}_{:.3f}_{:d}'.format(sampler_method,
                                                   self.dataset.config.in_radius,
                                                   self.dataset.config.first_subsampling_dl,
                                                   self.dataset.config.batch_num)
            batch_lim_dict[key] = float(self.dataset.batch_limit)
            with open(batch_lim_file, 'wb') as file:
                pickle.dump(batch_lim_dict, file)

            # Save neighb_limit dictionary
            for layer_ind in range(self.dataset.config.num_layers):
                dl = self.dataset.config.first_subsampling_dl * (2 ** layer_ind)
                if self.dataset.config.deform_layers[layer_ind]:
                    r = dl * self.dataset.config.deform_radius
                else:
                    r = dl * self.dataset.config.conv_radius
                key = '{:.3f}_{:.3f}'.format(dl, r)
                neighb_lim_dict[key] = self.dataset.neighborhood_limits[layer_ind]
            with open(neighb_lim_file, 'wb') as file:
                pickle.dump(neighb_lim_dict, file)


        print('Calibration done in {:.1f}s\n'.format(time.time() - t0))
        return


class ISPRSCustomBatch:
    """Custom batch definition with memory pinning for ISPRS"""

    def __init__(self, input_list):

        # Get rid of batch dimension
        input_list = input_list[0]

        # Number of layers
        L = (len(input_list) - 7) // 5

        # Extract input tensors from the list of numpy array
        ind = 0
        self.points = [torch.from_numpy(nparray) for nparray in input_list[ind:ind+L]]
        ind += L
        self.neighbors = [torch.from_numpy(nparray) for nparray in input_list[ind:ind+L]]
        ind += L
        self.pools = [torch.from_numpy(nparray) for nparray in input_list[ind:ind+L]]
        ind += L
        self.upsamples = [torch.from_numpy(nparray) for nparray in input_list[ind:ind+L]]
        ind += L
        self.lengths = [torch.from_numpy(nparray) for nparray in input_list[ind:ind+L]]
        ind += L
        self.features = torch.from_numpy(input_list[ind])
        ind += 1
        self.labels = torch.from_numpy(input_list[ind])
        ind += 1
        self.scales = torch.from_numpy(input_list[ind])
        ind += 1
        self.rots = torch.from_numpy(input_list[ind])
        ind += 1
        self.cloud_inds = torch.from_numpy(input_list[ind])
        ind += 1
        self.center_inds = torch.from_numpy(input_list[ind])
        ind += 1
        self.input_inds = torch.from_numpy(input_list[ind])

        return

    def pin_memory(self):
        """
        Manual pinning of the memory
        """

        self.points = [in_tensor.pin_memory() for in_tensor in self.points]
        self.neighbors = [in_tensor.pin_memory() for in_tensor in self.neighbors]
        self.pools = [in_tensor.pin_memory() for in_tensor in self.pools]
        self.upsamples = [in_tensor.pin_memory() for in_tensor in self.upsamples]
        self.lengths = [in_tensor.pin_memory() for in_tensor in self.lengths]
        self.features = self.features.pin_memory()
        self.labels = self.labels.pin_memory()
        self.scales = self.scales.pin_memory()
        self.rots = self.rots.pin_memory()
        self.cloud_inds = self.cloud_inds.pin_memory()
        self.center_inds = self.center_inds.pin_memory()
        self.input_inds = self.input_inds.pin_memory()

        return self

    def to(self, device):

        self.points = [in_tensor.to(device) for in_tensor in self.points]
        self.neighbors = [in_tensor.to(device) for in_tensor in self.neighbors]
        self.pools = [in_tensor.to(device) for in_tensor in self.pools]
        self.upsamples = [in_tensor.to(device) for in_tensor in self.upsamples]
        self.lengths = [in_tensor.to(device) for in_tensor in self.lengths]
        self.features = self.features.to(device)
        self.labels = self.labels.to(device)
        self.scales = self.scales.to(device)
        self.rots = self.rots.to(device)
        self.cloud_inds = self.cloud_inds.to(device)
        self.center_inds = self.center_inds.to(device)
        self.input_inds = self.input_inds.to(device)

        return self

    def unstack_points(self, layer=None):
        """Unstack the points"""
        return self.unstack_elements('points', layer)

    def unstack_neighbors(self, layer=None):
        """Unstack the neighbors indices"""
        return self.unstack_elements('neighbors', layer)

    def unstack_pools(self, layer=None):
        """Unstack the pooling indices"""
        return self.unstack_elements('pools', layer)

    def unstack_elements(self, element_name, layer=None, to_numpy=True):
        """
        Return a list of the stacked elements in the batch at a certain layer. If no layer is given, then return all
        layers
        """

        if element_name == 'points':
            elements = self.points
        elif element_name == 'neighbors':
            elements = self.neighbors
        elif element_name == 'pools':
            elements = self.pools[:-1]
        else:
            raise ValueError('Unknown element name: {:s}'.format(element_name))

        all_p_list = []
        for layer_i, layer_elems in enumerate(elements):

            if layer is None or layer == layer_i:

                i0 = 0
                p_list = []
                if element_name == 'pools':
                    lengths = self.lengths[layer_i+1]
                else:
                    lengths = self.lengths[layer_i]

                for b_i, length in enumerate(lengths):

                    elem = layer_elems[i0:i0 + length]
                    if element_name == 'neighbors':
                        elem[elem >= self.points[layer_i].shape[0]] = -1
                        elem[elem >= 0] -= i0
                    elif element_name == 'pools':
                        elem[elem >= self.points[layer_i].shape[0]] = -1
                        elem[elem >= 0] -= torch.sum(self.lengths[layer_i][:b_i])
                    i0 += length

                    if to_numpy:
                        p_list.append(elem.numpy())
                    else:
                        p_list.append(elem)

                if layer == layer_i:
                    return p_list

                all_p_list.append(p_list)

        return all_p_list


def ISPRSCollate(batch_data):
    return ISPRSCustomBatch(batch_data)

class ISPRSCustomBatchWeak:
    """Custom batch definition with memory pinning for ISPRS"""

    def __init__(self, input_list):

        # Get rid of batch dimension
        input_list = input_list[0]

        # Number of layers
        L = (len(input_list) - 8) // 5

        # Extract input tensors from the list of numpy array
        ind = 0
        self.points = [torch.from_numpy(nparray) for nparray in input_list[ind:ind+L]]
        ind += L
        self.neighbors = [torch.from_numpy(nparray) for nparray in input_list[ind:ind+L]]
        ind += L
        self.pools = [torch.from_numpy(nparray) for nparray in input_list[ind:ind+L]]
        ind += L
        self.upsamples = [torch.from_numpy(nparray) for nparray in input_list[ind:ind+L]]
        ind += L
        self.lengths = [torch.from_numpy(nparray) for nparray in input_list[ind:ind+L]]
        ind += L
        self.features = torch.from_numpy(input_list[ind])
        ind += 1
        self.labels = torch.from_numpy(input_list[ind])
        ind += 1
        self.scales = torch.from_numpy(input_list[ind])
        ind += 1
        self.rots = torch.from_numpy(input_list[ind])
        ind += 1
        self.cloud_inds = torch.from_numpy(input_list[ind])
        ind += 1
        self.center_inds = torch.from_numpy(input_list[ind])
        ind += 1
        self.input_inds = torch.from_numpy(input_list[ind])
        ind += 1
        self.points_weak_inds = [torch.from_numpy(nparray) for nparray in input_list[ind]]

        return

    def pin_memory(self):
        """
        Manual pinning of the memory
        """

        self.points = [in_tensor.pin_memory() for in_tensor in self.points]
        self.neighbors = [in_tensor.pin_memory() for in_tensor in self.neighbors]
        self.pools = [in_tensor.pin_memory() for in_tensor in self.pools]
        self.upsamples = [in_tensor.pin_memory() for in_tensor in self.upsamples]
        self.lengths = [in_tensor.pin_memory() for in_tensor in self.lengths]
        self.features = self.features.pin_memory()
        self.labels = self.labels.pin_memory()
        self.scales = self.scales.pin_memory()
        self.rots = self.rots.pin_memory()
        self.cloud_inds = self.cloud_inds.pin_memory()
        self.center_inds = self.center_inds.pin_memory()
        self.input_inds = self.input_inds.pin_memory()
        self.points_weak_inds = [in_tensor.pin_memory() for in_tensor in self.points_weak_inds]

        return self

    def to(self, device):

        self.points = [in_tensor.to(device) for in_tensor in self.points]
        self.neighbors = [in_tensor.to(device) for in_tensor in self.neighbors]
        self.pools = [in_tensor.to(device) for in_tensor in self.pools]
        self.upsamples = [in_tensor.to(device) for in_tensor in self.upsamples]
        self.lengths = [in_tensor.to(device) for in_tensor in self.lengths]
        self.features = self.features.to(device)
        self.labels = self.labels.to(device)
        self.scales = self.scales.to(device)
        self.rots = self.rots.to(device)
        self.cloud_inds = self.cloud_inds.to(device)
        self.center_inds = self.center_inds.to(device)
        self.input_inds = self.input_inds.to(device)
        self.points_weak_inds = [in_tensor.to(device) for in_tensor in self.points_weak_inds]

        return self

    def unstack_points(self, layer=None):
        """Unstack the points"""
        return self.unstack_elements('points', layer)

    def unstack_neighbors(self, layer=None):
        """Unstack the neighbors indices"""
        return self.unstack_elements('neighbors', layer)

    def unstack_pools(self, layer=None):
        """Unstack the pooling indices"""
        return self.unstack_elements('pools', layer)

    def unstack_elements(self, element_name, layer=None, to_numpy=True):
        """
        Return a list of the stacked elements in the batch at a certain layer. If no layer is given, then return all
        layers
        """

        if element_name == 'points':
            elements = self.points
        elif element_name == 'neighbors':
            elements = self.neighbors
        elif element_name == 'pools':
            elements = self.pools[:-1]
        else:
            raise ValueError('Unknown element name: {:s}'.format(element_name))

        all_p_list = []
        for layer_i, layer_elems in enumerate(elements):

            if layer is None or layer == layer_i:

                i0 = 0
                p_list = []
                if element_name == 'pools':
                    lengths = self.lengths[layer_i+1]
                else:
                    lengths = self.lengths[layer_i]

                for b_i, length in enumerate(lengths):

                    elem = layer_elems[i0:i0 + length]
                    if element_name == 'neighbors':
                        elem[elem >= self.points[layer_i].shape[0]] = -1
                        elem[elem >= 0] -= i0
                    elif element_name == 'pools':
                        elem[elem >= self.points[layer_i].shape[0]] = -1
                        elem[elem >= 0] -= torch.sum(self.lengths[layer_i][:b_i])
                    i0 += length

                    if to_numpy:
                        p_list.append(elem.numpy())
                    else:
                        p_list.append(elem)

                if layer == layer_i:
                    return p_list

                all_p_list.append(p_list)

        return all_p_list


def ISPRSCollateWeak(batch_data):
    return ISPRSCustomBatchWeak(batch_data)


# ----------------------------------------------------------------------------------------------------------------------
#
#           Debug functions
#       \*********************/


def debug_upsampling(dataset, loader):
    """Shows which labels are sampled according to strategy chosen"""


    for epoch in range(10):

        for batch_i, batch in enumerate(loader):

            pc1 = batch.points[1].numpy()
            pc2 = batch.points[2].numpy()
            up1 = batch.upsamples[1].numpy()

            print(pc1.shape, '=>', pc2.shape)
            print(up1.shape, np.max(up1))

            pc2 = np.vstack((pc2, np.zeros_like(pc2[:1, :])))

            # Get neighbors distance
            p0 = pc1[10, :]
            neighbs0 = up1[10, :]
            neighbs0 = pc2[neighbs0, :] - p0
            d2 = np.sum(neighbs0 ** 2, axis=1)

            print(neighbs0.shape)
            print(neighbs0[:5])
            print(d2[:5])

            print('******************')
        print('*******************************************')

    _, counts = np.unique(dataset.input_labels, return_counts=True)
    print(counts)


def debug_timing(dataset, loader):
    """Timing of generator function"""

    t = [time.time()]
    last_display = time.time()
    mean_dt = np.zeros(2)
    estim_b = dataset.config.batch_num
    estim_N = 0

    for epoch in range(10):

        for batch_i, batch in enumerate(loader):
            # print(batch_i, tuple(points.shape),  tuple(normals.shape), labels, indices, in_sizes)

            # New time
            t = t[-1:]
            t += [time.time()]

            # Update estim_b (low pass filter)
            estim_b += (len(batch.cloud_inds) - estim_b) / 100
            estim_N += (batch.features.shape[0] - estim_N) / 10

            # Pause simulating computations
            time.sleep(0.05)
            t += [time.time()]

            # Average timing
            mean_dt = 0.9 * mean_dt + 0.1 * (np.array(t[1:]) - np.array(t[:-1]))

            # Console display (only one per second)
            if (t[-1] - last_display) > -1.0:
                last_display = t[-1]
                message = 'Step {:08d} -> (ms/batch) {:8.2f} {:8.2f} / batch = {:.2f} - {:.0f}'
                print(message.format(batch_i,
                                     1000 * mean_dt[0],
                                     1000 * mean_dt[1],
                                     estim_b,
                                     estim_N))

        print('************* Epoch ended *************')

    _, counts = np.unique(dataset.input_labels, return_counts=True)
    print(counts)


def debug_show_clouds(dataset, loader):


    for epoch in range(10):

        clouds = []
        cloud_normals = []
        cloud_labels = []

        L = dataset.config.num_layers

        for batch_i, batch in enumerate(loader):

            # Print characteristics of input tensors
            print('\nPoints tensors')
            for i in range(L):
                print(batch.points[i].dtype, batch.points[i].shape)
            print('\nNeigbors tensors')
            for i in range(L):
                print(batch.neighbors[i].dtype, batch.neighbors[i].shape)
            print('\nPools tensors')
            for i in range(L):
                print(batch.pools[i].dtype, batch.pools[i].shape)
            print('\nStack lengths')
            for i in range(L):
                print(batch.lengths[i].dtype, batch.lengths[i].shape)
            print('\nFeatures')
            print(batch.features.dtype, batch.features.shape)
            print('\nLabels')
            print(batch.labels.dtype, batch.labels.shape)
            print('\nAugment Scales')
            print(batch.scales.dtype, batch.scales.shape)
            print('\nAugment Rotations')
            print(batch.rots.dtype, batch.rots.shape)
            print('\nModel indices')
            print(batch.model_inds.dtype, batch.model_inds.shape)

            print('\nAre input tensors pinned')
            print(batch.neighbors[0].is_pinned())
            print(batch.neighbors[-1].is_pinned())
            print(batch.points[0].is_pinned())
            print(batch.points[-1].is_pinned())
            print(batch.labels.is_pinned())
            print(batch.scales.is_pinned())
            print(batch.rots.is_pinned())
            print(batch.model_inds.is_pinned())

            show_input_batch(batch)

        print('*******************************************')

    _, counts = np.unique(dataset.input_labels, return_counts=True)
    print(counts)


def debug_batch_and_neighbors_calib(dataset, loader):
    """Timing of generator function"""

    t = [time.time()]
    last_display = time.time()
    mean_dt = np.zeros(2)

    for epoch in range(10):

        for batch_i, input_list in enumerate(loader):
            # print(batch_i, tuple(points.shape),  tuple(normals.shape), labels, indices, in_sizes)

            # New time
            t = t[-1:]
            t += [time.time()]

            # Pause simulating computations
            time.sleep(0.01)
            t += [time.time()]

            # Average timing
            mean_dt = 0.9 * mean_dt + 0.1 * (np.array(t[1:]) - np.array(t[:-1]))

            # Console display (only one per second)
            if (t[-1] - last_display) > 1.0:
                last_display = t[-1]
                message = 'Step {:08d} -> Average timings (ms/batch) {:8.2f} {:8.2f} '
                print(message.format(batch_i,
                                     1000 * mean_dt[0],
                                     1000 * mean_dt[1]))

        print('************* Epoch ended *************')

    _, counts = np.unique(dataset.input_labels, return_counts=True)
    print(counts)
