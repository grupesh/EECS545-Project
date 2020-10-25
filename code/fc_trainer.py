import os
import torch
import time
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import save_checkpoint
import torch.nn as nn
from dataset import anomaly_detect_loader, fcnet_loader
class fcnet_trainer:
    def __init__(self,
                 model,
                 norm_path=None,
                 abnorm_path=None,
                 learning_rate=2e-3,
                 num_epochs=20,
                 batch_size=8,
                 ckpt_dir = ''
                 ):
        self.model = model
        self.num_epochs = num_epochs
        self.norm_path = norm_path
        self.abnorm_path = abnorm_path
        self.ckpt_dir = ckpt_dir
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            learning_rate)  # leave betas and eps by default
        self.lr_scheduler = ReduceLROnPlateau(self.optimizer, verbose=True, factor=0.2)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
    def my_criterion(self,
                    norm_scores,
                    abnorm_scores,
                    reg1=8e-5,
                    reg2=8e-5):
        num_segments = 32
        norm_scores_bat = norm_scores.reshape((self.batch_size, num_segments))
        abnorm_scores_bat = abnorm_scores.reshape((self.batch_size, num_segments))
        temp = 1 - abnorm_scores_bat.max(dim=1)[0] + norm_scores_bat.max(dim=1)[0]  # shape would be batch_size
        loss_term1 = (1 / self.batch_size) * torch.sum(torch.max(torch.zeros(self.batch_size).to(self.device), temp))
        loss_term2 = torch.mean(
            torch.sum((abnorm_scores_bat[:, :num_segments - 1] - abnorm_scores_bat[:, 1:]) ** 2, 1))
        loss_term3 = torch.mean(torch.sum(abnorm_scores_bat, 1))
        loss = loss_term1 + reg1 * loss_term2 + reg2 * loss_term3
        return loss
    def train(self):
        Loss_history = []
        self.model.train()
        for i in range(self.num_epochs):
            start_t = time.time()
            self.optimizer.zero_grad()
            train_loader = fcnet_loader(norm_path= self.norm_path,
                                       abnorm_path= self.abnorm_path,
                                       test_path= None,
                                       is_train= True,
                                       batch_size= self.batch_size)
            train_data = train_loader.load()
            # only one batch
            for iter_num, batch in enumerate(train_data):
                for key in batch:
                    # print('key: {}, maximum value: {}'.format(key, torch.max(batch[key]).item()))
                    batch[key] = batch[key].to(self.device)
                norm_scores = self.model.forward(batch['normal'])
                abnorm_scores = self.model.forward(batch['abnormal'])
                loss = self.my_criterion(norm_scores= norm_scores,
                                         abnorm_scores = abnorm_scores)
                Loss_history.append(loss.item())
                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step(loss)
            end_t = time.time()
            print('(Epoch {} / {}) train loss: {:.4f} time per epoch: {:.1f}s current lr: {}'.format(
                    i + 1, self.num_epochs, Loss_history[i], end_t - start_t,
                    self.optimizer.param_groups[0]['lr']))
            if i == np.argmin(Loss_history):
                print('The current model is the best model! Save it!')
                if torch.cuda.is_available():
                    if torch.cuda.device_count() > 1:
                        save_checkpoint(self.model.module.state_dict(), is_best=True,
                                    checkpoint_dir= self.ckpt_dir)
                    else:
                        save_checkpoint(self.model.state_dict(), is_best=True,
                                        checkpoint_dir= self.ckpt_dir)
                else:
                    save_checkpoint(self.model.state_dict(), is_best=True,
                                    checkpoint_dir= self.ckpt_dir)
            if (i + 1) % 10 == 0:
                print('Save the current model to checkpoint!')
                if torch.cuda.is_available():
                    if torch.cuda.device_count() > 1:
                        save_checkpoint(self.model.module.state_dict(), is_best=False,
                                        checkpoint_dir= self.ckpt_dir,
                                        name = str(i).zfill(5))
                    else:
                        save_checkpoint(self.model.state_dict(), is_best=False,
                                        checkpoint_dir= self.ckpt_dir,
                                        name=str(i).zfill(5))
                else:
                    save_checkpoint(self.model.state_dict(), is_best=False,
                                    checkpoint_dir= self.ckpt_dir,
                                    name=str(i).zfill(5))
                torch.save(Loss_history, os.path.join(self.ckpt_dir, 'train_loss.pt'))
            print('(epoch {} / {})'.format(i + 1, self.num_epochs))












