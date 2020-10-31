import torch
from torchsummary import summary
from torchvision import datasets, models
import torch.nn as nn
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
import os
"""
Input: 
`norm_path`: path to normal video (cleaned) list, default = None.
`abnorm_path`: path to abnormal video (cleaned) list, default = None.
`is_train`: default = true, when testing, set this to false.
`batch_size`: batch size, default = 30.
`segment_num`: number of segmentations in each video, default = 32.

Output:
`Dataloader`: A dataloader class having the following key(s):
if training:
  Dataloader['norm_video']: (`batch_size` * 32) * 4096
  Dataloder['abnorm_video']: (`batch_size` * 32) * 4096
if testing:
  Dataloder['test_video']: (`batch_size` * 32) * 4096

To do:
If training:
At each iteration, randomly choose `batch_size` normal videos and `batch_size` abnormal videos,
then create a dataloader having 2 keys corresponding to these normal / abnormal videos (2 * `batch_size` videos in total).
If testing:
Randomly choose `batch_size` test videos.
"""

# update I do not use the test path to load features, rather I load stored list which has segment level annotations too
# I added another flag is_vald which define validation loader

class anomaly_detect_loader(data.Dataset):
    def __init__(self,
                 norm_path=None,
                 abnorm_path=None,
                 is_train=True,
                 is_vald=False,
                 batch_size=30,
                 verb = True,
                 root_path=None):
        super(anomaly_detect_loader, self).__init__()
        self.is_train = is_train
        self.is_vald = is_vald
        self.num_segment = 32
        self.batch_size = batch_size
        self.verb = verb
        self.root_path = root_path
        if self.is_train:
            self.norm_list = np.loadtxt(norm_path, dtype= str)
            self.abnorm_list = np.loadtxt(abnorm_path, dtype = str)
            self.norm_choice = np.random.choice(len(self.norm_list), self.batch_size) # uniformly sample
            self.abnorm_choice = np.random.choice(len(self.abnorm_list), self.batch_size) # uniformly sample
            self.meta = {'normal': [], 'abnormal': []}
        else:
            if is_vald:
                self.vald_list = torch.load(os.path.join(os.getcwd(),'..','saved_data/validation_annotations.pt'))
                self.norm_list = [item for item in self.vald_list if item['category'] == 'Testing_Normal_Videos_Anomaly']
                # print(self.norm_list)
                self.abnorm_list = [item for item in self.vald_list if item not in self.norm_list]
                self.norm_choice = np.random.choice(len(self.norm_list), self.batch_size)  # uniformly sample
                self.abnorm_choice = np.random.choice(len(self.abnorm_list), self.batch_size)  # uniformly sample
                #self.vald_choice = np.random.choice(len(self.vald_list), self.batch_size)
                self.meta = {'normal': [], 'abnormal': [], 'labels_norm': [], 'labels_abnorm': []}
            else:
                self.test_list = torch.load(os.path.join(os.getcwd(),'..','saved_data/test_annotations.pt'))
                # print(self.test_list)
                self.test_choice = np.random.choice(len(self.test_list), self.batch_size)
                self.meta = {'test': [], 'labels': []}
        self.load_features()

    def load_features(self):
        if self.root_path is not None:
            root_path = self.root_path
        else:
            root_path = os.path.abspath(os.path.join(os.getcwd(), ".."))
        # root_path = 'G:\Shared drives\EECS 545 - ML Project\data'
        if self.is_train:
            for i in range(len(self.norm_choice)):
                norm_feature = root_path + '/features/' + self.norm_list[self.norm_choice[i]].split('.')[0] + '.pt'
                abnorm_feature = root_path + '/features/' + self.abnorm_list[self.abnorm_choice[i]].split('.')[0] + '.pt'
                if self.verb:
                    print('loading normal features from: ', norm_feature)
                    print('loading abnormal features from: ', abnorm_feature)
                self.meta['normal'].append(torch.load(norm_feature))
                self.meta['abnormal'].append(torch.load(abnorm_feature))
            self.meta['normal'] = torch.stack(self.meta['normal']).reshape(-1, self.meta['normal'][0].shape[-1])
            self.meta['abnormal'] = torch.stack(self.meta['abnormal']).reshape(-1, self.meta['abnormal'][0].shape[-1])
            if self.verb:
                print('shape of normal features: ', self.meta['normal'].shape)
                print('shape of abnormal features: ', self.meta['abnormal'].shape)
        elif self.is_vald:
            # print(len(self.norm_choice))
            for i in range(len(self.norm_choice)):
                category = self.abnorm_list[self.abnorm_choice[i]]['category']
                filename_split_norm = self.norm_list[self.norm_choice[i]]['file_path'].split('/Videos/')
                filename_split_abnorm = self.abnorm_list[self.abnorm_choice[i]]['file_path'].split('/Videos/')
                if self.root_path is not None:
                    root_path = self.root_path
                else:
                    root_path = filename_split_norm[0]
                # print(filename_split_norm)
                #if category == 'Normal':
                filename_norm = root_path + '/features/' + filename_split_norm[1]
                filename_abnorm = root_path + '/features/' + category + '/' + category + filename_split_abnorm[1].split(category)[2]
                #vald_feature = filename.split('.')[0] + '.pt'
                norm_feature = filename_norm.split('.')[0] + '.pt'
                abnorm_feature = filename_abnorm.split('.')[0] + '.pt'
                if self.verb:
                    print('loading normal features from: ', norm_feature)
                    print('loading abnormal features from: ', abnorm_feature)
                anomaly_segments_norm = self.norm_list[self.norm_choice[i]]['anomaly_frames']
                vald_label_norm = torch.zeros(32,1)
                vald_label_norm[anomaly_segments_norm] = 1
                anomaly_segments_abnorm = self.abnorm_list[self.abnorm_choice[i]]['anomaly_frames']
                vald_label_abnorm = torch.zeros(32,1)
                vald_label_abnorm[anomaly_segments_abnorm] = 1
                self.meta['normal'].append(torch.load(norm_feature))
                self.meta['abnormal'].append(torch.load(abnorm_feature))
                self.meta['labels_norm'].append(vald_label_norm)
                self.meta['labels_abnorm'].append(vald_label_abnorm)
            self.meta['normal'] = torch.stack(self.meta['normal']).reshape(-1, self.meta['normal'][0].shape[-1])
            self.meta['abnormal'] = torch.stack(self.meta['abnormal']).reshape(-1, self.meta['abnormal'][0].shape[-1])
            #self.meta['vald'] = torch.stack(self.meta['vald']).reshape(-1, self.meta['vald'][0].shape[-1])
            self.meta['labels_norm'] = torch.stack(self.meta['labels_norm']).reshape(-1, self.meta['labels_norm'][0].shape[-1])
            self.meta['labels_abnorm'] = torch.stack(self.meta['labels_abnorm']).reshape(-1, self.meta['labels_abnorm'][0].shape[-1])
            if self.verb:
                print('shape of validation features: ', self.meta['normal'].shape)
                print('shape of validation labels: ', self.meta['labels_norm'].shape)
        else:
            for i in range(len(self.test_choice)):
                category = self.test_list[self.test_choice[i]]['category']
                filename_split = self.test_list[self.test_choice[i]]['file_path'].split('/Videos/')
                if self.root_path is not None:
                    root_path = self.root_path
                else:
                    root_path = filename_split[0]
                # print(filename_split)
                if category == 'Testing_Normal_Videos_Anomaly':
                    filename = root_path + '/features/' + filename_split[1]
                else:
                    filename = root_path + '/features/' + category + '/' + category + \
                               filename_split[1].split(category)[2]

                test_feature = filename.split('.')[0] + '.pt'
                if self.verb:
                    print('loading test feature from: ', test_feature)
                anomaly_segments = self.test_list[self.test_choice[i]]['anomaly_frames']
                test_label = torch.zeros(32,1)
                test_label[anomaly_segments] = 1
                self.meta['test'].append(torch.load(test_feature))
                self.meta['labels'].append(test_label)
            self.meta['test'] = torch.stack(self.meta['test']).reshape(-1, self.meta['test'][0].shape[-1])
            self.meta['labels'] = torch.stack(self.meta['labels']).reshape(-1, self.meta['labels'][0].shape[-1])
            if self.verb:
                print('shape of test features: ', self.meta['test'].shape)
                print('shape of test labels: ', self.meta['labels'].shape)

    def __getitem__(self, index):
        Meta = {}
        if self.is_train:
            Meta['normal'] = self.meta['normal'][index, :]
            Meta['abnormal'] = self.meta['abnormal'][index, :]
        elif self.is_vald:
            Meta['normal'] = self.meta['normal'][index,:]
            Meta['labels_norm'] = self.meta['labels_norm'][index,:]
            Meta['abnormal'] = self.meta['abnormal'][index, :]
            Meta['labels_abnorm'] = self.meta['labels_abnorm'][index, :]
        else:
            Meta['test'] = self.meta['test'][index, :]
            Meta['labels'] = self.meta['labels'][index,:]

        return Meta

    def __len__(self):
        return self.batch_size * self.num_segment
class fcnet_loader():
    def __init__(self,
                 norm_path=None,
                 abnorm_path=None,
                 is_train=True,
                 is_vald=False,
                 batch_size=30,
                 verb = True,
                 root_path = None):
        self.norm_path = norm_path
        self.abnorm_path = abnorm_path
        self.is_train = is_train
        self.is_vald = is_vald
        self.batch_size = batch_size
        self.verb = verb
        self.root_path = root_path
    def load(self):
        loader = anomaly_detect_loader(norm_path=self.norm_path,
                                       abnorm_path=self.abnorm_path,
                                       is_train=self.is_train,
                                       is_vald=self.is_vald,
                                       batch_size=self.batch_size,
                                       verb = self.verb,
                                       root_path= self.root_path)
        train_test_loader = data.DataLoader(loader,
                                            batch_size=self.batch_size * loader.num_segment,
                                            shuffle=False,
                                            num_workers=8,
                                            pin_memory=True)
        return train_test_loader


if __name__ == '__main__':
    norm_path = os.path.join(os.getcwd(), "..", 'Anomaly_Detection_splits', 'Normal_Train_clean.txt')
    abnorm_path = os.path.join(os.getcwd(), "..", 'Anomaly_Detection_splits', 'Anomaly_Train_clean.txt')
    # test path not required anymore
    # I'm still passing it as argument, not being used
    # Zongyu, review this code and if you agree I will remove test_path
    # Zongyu: test_path is removed!
    # test_path = os.path.join(os.getcwd(), "..", 'Anomaly_Detection_splits', 'Anomaly_Test.txt')
    to_train = True
    to_vald = False
    to_test = False
    if to_train:
        train_loader = fcnet_loader(norm_path = norm_path,
                                   abnorm_path = abnorm_path,
                                   is_train= True,
                                    is_vald=False,
                                   batch_size= 30,
                                    verb=True,
                                    root_path='G:/Shared drives/EECS 545 - ML Project/data')
        train_data = train_loader.load()
        for batch in train_data:
            print(batch['normal'].shape)
    if to_vald:
        vald_loader = fcnet_loader(norm_path=norm_path,
                                abnorm_path=abnorm_path,
                                is_train=False,
                                is_vald=True,
                                batch_size=5,
                                verb= False,
                                   root_path=None)
        vald_data = vald_loader.load()
        for batch in vald_data:
            print(batch['normal'].shape)
            print(batch['labels_norm'].shape)
    if to_test:
        test_loader = fcnet_loader(norm_path=norm_path,
                                   abnorm_path=abnorm_path,
                                   is_train=False,
                                   is_vald=False,
                                   batch_size=30,
                                   root_path=None)
        test_data = test_loader.load()
        for batch in test_data:
            print(batch['test'].shape)
            print(batch['labels'].shape)

