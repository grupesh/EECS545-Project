import os
import random
from gen_test_annotations import gen_annotations
import torch

def define_validation_set(f_path):
  """
  Input:
  'f_path' : location of test annotations file
  Output:
  'vald_set' : list of annotated samples chosen in validation set,
  we choose 1/5 of normal videos and 1/4 of anomaly videos from test set in validation set
  Also saves this list as txt file and returns it location
  """
  #f_path = '/content/drive/Shared drives/EECS 545 - ML Project/data/Temporal_Anomaly_Annotation_for_Testing_Videos.txt'
  f_obj= open(f_path,'r')
  alllines = f_obj.readlines()
  normal_set = [lines for lines in alllines if lines.startswith('Normal')]
  anomaly_set = [lines for lines in alllines if lines not in normal_set]
  num_normal_test = len(normal_set)
  num_normal_vald = num_normal_test // 5
  num_anomaly_test = len(anomaly_set)
  num_anomaly_vald = num_anomaly_test // 4
  val_set_normal = random.sample(normal_set,num_normal_vald)
  val_set_anomaly = random.sample(anomaly_set,num_anomaly_vald)
  vald_set = val_set_normal + val_set_anomaly
  vald_set = random.sample(vald_set, len(vald_set))
  filename = os.path.join(os.getcwd(),'..','Anomaly_Detection_splits/Temporal_annotations_for_vald.txt')
  if not os.path.exists(filename):
    with open(filename, 'w') as f:
      for item in vald_set:
        f.write("%s" % item)
  return vald_set, filename

if __name__ == '__main__':
    test_path = os.path.join(os.getcwd(),'..','Anomaly_Detection_splits/Temporal_Anomaly_Annotation_for_Testing_Videos.txt')
    vald_set, filename = define_validation_set(test_path)
    print('Samples defined, generating list for easy access')
    vald_set_final = gen_annotations(filename)
    write_path = os.path.join(os.getcwd(),'..','saved_data/validation_annotations.pt')
    torch.save(vald_set_final,write_path)
    print('Generated list saved!')