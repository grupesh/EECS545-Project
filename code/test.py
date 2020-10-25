from fc_trainer import fcnet_trainer
from model import FC_net
import torch
import torch.nn as nn
import argparse
import os
import time

test_path = os.path.join(os.getcwd(), "..", 'Anomaly_Detection_splits', 'Anomaly_Test.txt')
parser = argparse.ArgumentParser()
parser.add_argument('--batch', type=int, help='the test batch size', default=1)
parser.add_argument('--is_best', action='store_true', help='load best / last checkpoint')
parser.add_argument("--mode", default='client')
parser.add_argument("--port", default=50093)
args = parser.parse_args()
