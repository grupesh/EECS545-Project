from backbone import I3D_backbone
import numpy as np
import torch
import torch.nn as nn
import cv2
import os
import torchvision.transforms as transforms
import torchvision.datasets
import torch.utils.data as data
import matplotlib.pyplot as plt
import gc
def load_pretrain():
    model = I3D_backbone()
    pretrained_model = torch.load('./model/model_rgb.pth')
    for i, key in enumerate(pretrained_model):
        if i == len(pretrained_model) - 2:
            break
        for backbone_key in model.state_dict():
            if backbone_key == key:
                matched = backbone_key
                model.state_dict()[matched] = pretrained_model[matched]
                break
    return model
class video_feature(data.Dataset):
    def __init__(self,
                 video_path = None,
                 num_segment = 32,
                 normalize= True):
        super(video_feature, self).__init__()
        self.num_segment = num_segment
        if normalize:
            transform_list = [
                transforms.ToTensor(),  # convert image to PyTorch tensor
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Imagenet mean and std
            ]
        else:
            transform_list = [transforms.ToTensor()]
        self.transform = transforms.Compose(transform_list)
        self.vid = self.load_video_single(video_path)

    def __getitem__(self, item):
        return self.transform(self.vid[item, :, :, :] / 255)
    def __len__(self):
        return self.vid.shape[0]
    def load_video_single(self, path):

        cap = cv2.VideoCapture(path)
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if frameCount > 36000: # we don't process videos larger than 20 mins
            raise ValueError('We dont process videos longer than 20 mins')
        else:
            fc = 0
            ret = True
            vid = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))
            while (fc < frameCount and ret):
                ret, vid[fc] = cap.read()
                fc += 1
            vid = vid[:(vid.shape[0] // self.num_segment) * self.num_segment,:,:,:]
            return vid

if __name__ == '__main__':
    num_segment = 32
    num_frame = 16
    if not os.path.exists('../features'):
        os.mkdir('../features')
    path_list = []
    file_list = []
    for root, _, files in os.walk('../Videos'):
        path_list.append(root)
        files = [fi for fi in files if fi.endswith(".mp4")]
        # print(files)
        file_list.append(files)
    model = load_pretrain()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        if gpu_count > 1:
            print("There are", torch.cuda.device_count(), "GPUs!")
            devices = [0, 1]
            model = nn.DataParallel(model, device_ids=devices)
            print("But Let's use", len(devices), "GPUs!")
    model = model.float().to(device)
    path_num = len(path_list) - 1
    for i in range(path_num):
        video_files = file_list[i + 1]
        print('load data from {}'.format(path_list[i + 1]))
        if not os.path.exists('../features/' + path_list[i + 1].split('/')[-1]):
            os.mkdir('../features/' + path_list[i + 1].split('/')[-1])
        for v in video_files:
            print('processing: ', v)
            try:
                train_data = video_feature(video_path= os.path.join(path_list[i + 1], v), num_segment = num_segment, normalize= False)
                loader = data.DataLoader(train_data, batch_size= len(train_data), num_workers= 8, shuffle=False)
                with torch.no_grad():
                    running_out = []
                    for iter_num, batch in enumerate(loader): # only one batch
                        perm_batch = torch.reshape(batch, [num_segment, int(len(train_data) / num_segment), batch.shape[1], batch.shape[2], batch.shape[3]]).permute(0, 2, 1, 3, 4)
                        for r in range(int(perm_batch.shape[2] / num_frame) + 1):
                            if (r + 1) * num_frame <= perm_batch.shape[2]:
                                perm_frame = perm_batch[:, :, r * num_frame:(r + 1) * num_frame, :, :]
                            else:
                                perm_frame = perm_batch[:, :, perm_batch.shape[2] - num_frame:, :, :]
                            perm_frame = perm_frame.float().to(device)
                            out_frame = model.forward(perm_frame)
                            norm_out_frame = out_frame / torch.norm(out_frame, dim = 1, keepdim= True)
                            running_out.append(norm_out_frame)
                        mean_out = sum(running_out) / len(running_out)
                    # unfold_batch = perm_batch.unfold(2, num_frame, int(num_frame / 2)).permute(0, 2, 1, 5, 3, 4)
                    # final_batch = torch.reshape(unfold_batch, [-1, unfold_batch.shape[2], unfold_batch.shape[3], unfold_batch.shape[4], unfold_batch.shape[5]])
                    # final_batch = final_batch.float().to(device)
                    # out = model.forward(final_batch)
                    # norm_out = out / torch.norm(out, dim=1, keepdim=True)
                    # mean_out = torch.mean(torch.reshape(norm_out, [num_segment, -1, norm_out.shape[1]]), dim = 1)
                        torch.save(mean_out.cpu(), '../features/' + path_list[i + 1].split('/')[-1] + '/' + v.split('.')[0] + '.pt')
            except ValueError as err:
                print('{} cannot be processed due to {}!'.format(v, err))
                continue
            del train_data
            del loader
            gc.collect()