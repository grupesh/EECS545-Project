from fc_trainer import fcnet_trainer
from model import FC_net
import torch
import torch.nn as nn
import argparse
import os
norm_path = os.path.join(os.getcwd(), "..", 'Anomaly_Detection_splits', 'Normal_Train_clean.txt')
abnorm_path = os.path.join(os.getcwd(), "..", 'Anomaly_Detection_splits', 'Anomaly_Train_clean.txt')
parser = argparse.ArgumentParser()
parser.add_argument('--batch', type=int, help='the train batch size', default=1)
parser.add_argument('--batch_vald', type=int, help='the validation batch size', default=1)
parser.add_argument('--lr', type=float, help='the learning rate', default=2e-3)
parser.add_argument('--epochs', type=int, help='training epochs', default=50)
parser.add_argument('--continue_train', action='store_true', help='load checkpoint and continue training')
parser.add_argument("--mode", default='client')
parser.add_argument("--port", default=50093)
args = parser.parse_args()

class train_fcnet():
    def __init__(self,
                 batch_size,
                 batch_size_vald,
                 learning_rate,
                 num_epochs,
                 continue_train):
        # self.result_dir = os.path.join(os.getcwd(), '..', 'result')
        self.ckpt_dir = os.path.join(os.getcwd(), '..', 'checkpoint')
        # if not os.path.exists(self.result_dir):
            # os.mkdir(self.result_dir)
        if not os.path.exists(self.ckpt_dir):
            os.mkdir(self.ckpt_dir)
        self.model = FC_net()
        # print(continue_train)
        if continue_train:
            latest_ckpt = max(self.all_files_under(self.ckpt_dir), key=os.path.getmtime)
            try:
                self.model.load_state_dict(torch.load(latest_ckpt))
                print('load pre-trained model successful, continue training!')
            except:
                raise IOError(f"Checkpoint '{latest_ckpt}' load failed! ")

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            if gpu_count > 1:
                print("There are", torch.cuda.device_count(), "GPUs!")
                devices = [0, 1]
                self.model = nn.DataParallel(self.model, device_ids= devices)
                print("But Let's use", len(devices), "GPUs!")
        self.model.to(self.device)
        self.trainer = fcnet_trainer(model = self.model,
                                    norm_path= norm_path,
                                    abnorm_path= abnorm_path,
                                    learning_rate= learning_rate,
                                    num_epochs= num_epochs,
                                    batch_size= batch_size,
                                     batch_size_vald = batch_size_vald,
                                    ckpt_dir= self.ckpt_dir)
    def train(self):
        self.trainer.train()
    def all_files_under(self, path):
        """Iterates through all files that are under the given path."""
        for cur_path, dirnames, filenames in os.walk(path):
            for filename in filenames:
                yield os.path.join(cur_path, filename)
def main():
    fcnet = train_fcnet(batch_size= args.batch,
                        batch_size_vald=args.batch_vald,
                        learning_rate= args.lr,
                        num_epochs= args.epochs,
                        continue_train= args.continue_train)
    fcnet.train()
if __name__ == '__main__':
    main()

