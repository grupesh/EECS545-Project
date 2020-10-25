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
`test_path`: path to test video list, default = None.
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


class anomaly_detect_loader(data.Dataset):
    def __init__(self,
                 norm_path=None,
                 abnorm_path=None,
                 test_path=None,
                 is_train=True,
                 batch_size=30):
        super(anomaly_detect_loader, self).__init__()
        self.is_train = is_train
        self.num_segment = 32
        self.batch_size = batch_size
        if self.is_train:
            self.norm_list = np.loadtxt(norm_path, dtype= str)
            self.abnorm_list = np.loadtxt(abnorm_path, dtype = str)
            self.norm_choice = np.random.choice(len(self.norm_list), self.batch_size) # uniformly sample
            self.abnorm_choice = np.random.choice(len(self.abnorm_list), self.batch_size) # uniformly sample
            self.meta = {'normal': [], 'abnormal': []}
        else:
            self.test_list = np.loadtxt(test_path, dtype=str)
            self.test_choice = np.random.choice(len(self.test_list), self.batch_size)
            self.meta = {'test': []}
        self.load_features()

    def load_features(self):
        root_path = os.path.abspath(os.path.join(os.getcwd(), ".."))
        if self.is_train:
            for i in range(len(self.norm_choice)):
                norm_feature = root_path + '/features/' + self.norm_list[self.norm_choice[i]].split('.')[0] + '.pt'
                abnorm_feature = root_path + '/features/' + self.abnorm_list[self.abnorm_choice[i]].split('.')[0] + '.pt'
                print('loading normal features from: ', norm_feature)
                print('loading abnormal features from: ', abnorm_feature)
                self.meta['normal'].append(torch.load(norm_feature))
                self.meta['abnormal'].append(torch.load(abnorm_feature))
            self.meta['normal'] = torch.stack(self.meta['normal']).reshape(-1, self.meta['normal'][0].shape[-1])
            self.meta['abnormal'] = torch.stack(self.meta['abnormal']).reshape(-1, self.meta['abnormal'][0].shape[-1])
            print('shape of normal features: ', self.meta['normal'].shape)
            print('shape of abnormal features: ', self.meta['abnormal'].shape)
        else:
            for i in range(len(self.test_choice)):
                test_feature = root_path + '/features/' + self.test_list[self.test_choice[i]].split('.')[0] + '.pt'
                print('loading test feature from: ', test_feature)
                self.meta['test'].append(torch.load(test_feature))
            self.meta['test'] = torch.stack(self.meta['test']).reshape(-1, self.meta['test'][0].shape[-1])
            print('shape of test features: ', self.meta['test'].shape)

    def __getitem__(self, index):
        Meta = {}
        if self.is_train:
            Meta['normal'] = self.meta['normal'][index, :]
            Meta['abnormal'] = self.meta['abnormal'][index, :]
        else:
            Meta['test'] = self.meta['test'][index, :]

        return Meta

    def __len__(self):
        return self.batch_size * self.num_segment
class fcnet_loader():
    def __init__(self,
                 norm_path=None,
                 abnorm_path=None,
                 test_path=None,
                 is_train=True,
                 batch_size=30):
        self.norm_path = norm_path
        self.abnorm_path = abnorm_path
        self.test_path = test_path
        self.is_train = is_train
        self.batch_size = batch_size
    def load(self):
        loader = anomaly_detect_loader(norm_path=self.norm_path,
                                       abnorm_path=self.abnorm_path,
                                       test_path= self.test_path,
                                       is_train=self.is_train,
                                       batch_size=self.batch_size)
        train_test_loader = data.DataLoader(loader,
                                            batch_size=self.batch_size * loader.num_segment,
                                            shuffle=False,
                                            num_workers=8,
                                            pin_memory=True)
        return train_test_loader


if __name__ == '__main__':
    norm_path = os.path.join(os.getcwd(), "..", 'Anomaly_Detection_splits', 'Normal_Train_clean.txt')
    abnorm_path = os.path.join(os.getcwd(), "..", 'Anomaly_Detection_splits', 'Anomaly_Train_clean.txt')
    test_path = os.path.join(os.getcwd(), "..", 'Anomaly_Detection_splits', 'Anomaly_Test.txt')
    train_loader = fcnet_loader(norm_path = norm_path,
                               abnorm_path = abnorm_path,
                               test_path = test_path,
                               is_train= True,
                               batch_size= 30)
    train_data = train_loader.load()
    for batch in train_data:
        print(batch['normal'].shape)
