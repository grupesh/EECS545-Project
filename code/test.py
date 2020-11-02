from fc_trainer import fcnet_trainer
from dataset import fcnet_loader, anomaly_detect_loader
from model import FC_net
from utils import load_checkpoint
import torch
import torch.nn as nn
import argparse
import os
from sklearn import metrics
import numpy as np
import time

test_path = os.path.join(os.getcwd(), "..", 'Anomaly_Detection_splits', 'Anomaly_Test.txt')
parser = argparse.ArgumentParser()
parser.add_argument('--batch', type=int, help='the test batch size', default=1)
parser.add_argument('--is_best', action='store_true', help='load best / last checkpoint')
parser.add_argument("--mode", default='client')
parser.add_argument("--port", default=50093)
parser.add_argument('--ckpt_file', type=str, help = 'checkpoint file name')
parser.add_argument('--root_path', type=str, help = 'root path of features folder')
args = parser.parse_args()

class test_fcnet:
    def __init__(self,
                 batch_size=8,
                 ckpt_file = '',
                 root_path=None
                 ):
        self.batch_size = batch_size
        self.ckpt_dir = self.ckpt_dir = os.path.join(os.getcwd(), '..', 'checkpoint')
        self.ckpt_file = ckpt_file
        self.root_path = root_path

        self.model = FC_net()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            if gpu_count > 1:
                print("There are", torch.cuda.device_count(), "GPUs!")
                devices = [0, 1]
                self.model = nn.DataParallel(self.model, device_ids=devices)
                print("But Let's use", len(devices), "GPUs!")
        self.model.to(self.device)

    def my_accuracy(self,
                    scores=None,
                    labels=None):
        fpr, tpr, threshold = metrics.roc_curve(labels.detach().cpu().numpy(), scores.detach().cpu().numpy())
        optim_idx = np.argmax(tpr - fpr)
        optim_threshold = threshold[optim_idx]
        #print(scores.shape)
        #print(labels.shape)
        roc_auc = metrics.roc_auc_score(labels.detach().cpu().numpy(), scores.detach().cpu().numpy())
        optim_fpr = fpr[optim_idx]
        optim_tpr = tpr[optim_idx]
        return roc_auc, optim_threshold, optim_fpr, optim_tpr

    def test(self):
        auc_list = []
        optim_thresh_list = []
        optim_fpr_list = []
        optim_tpr_list = []
        checkpoint_file = os.path.join(self.ckpt_dir,self.ckpt_file)
        print(checkpoint_file)
        state = torch.load(checkpoint_file, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

        print('Running eval on test set')
        test_loader = fcnet_loader(norm_path=None,
                                   abnorm_path=None,
                                   is_train=False,
                                   is_vald=False,
                                   batch_size=self.batch_size,
                                   verb= True,
                                   root_path = self.root_path)
        test_data = test_loader.load()
        for iter_num, batch in enumerate(test_data):
            self.model.eval()
            for key in batch:
                # print('key: {}, maximum value: {}'.format(key, torch.max(batch[key]).item()))
                batch[key] = batch[key].to(self.device)
            scores = self.model.forward(batch['test']).reshape(-1)
            labels = batch['labels'].reshape(-1)
            print(labels.shape)
            auc,optim_thresh,optim_fpr,optim_tpr = self.my_accuracy(scores = scores, labels = labels)
            auc_list.append(auc)
            optim_thresh_list.append(optim_thresh)
            optim_fpr_list.append(optim_fpr)
            optim_tpr_list.append(optim_tpr)
            print('Batch : {} , auc score : {:.3f}'.format(iter_num, auc_list[iter_num]))

        result_dir = os.path.join(os.getcwd(),'..','results')
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)
        torch.save({'auc_vals' : auc_list,
                    'optim_threshold' : optim_thresh_list,
                    'optim_tpr' : optim_tpr_list,
                    'optim_fpr' : optim_fpr_list},os.path.join(result_dir,'result_dict.pt'))

def main():
    fcnet = test_fcnet(batch_size= args.batch,
                        ckpt_file= args.ckpt_file,
                        root_path = args.root_path)
    fcnet.test()
if __name__ == '__main__':
    main()