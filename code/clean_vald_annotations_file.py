import os
import numpy as np

def clean_labels():
    # load test clean normal and anomaly videos
    normal_file_path = os.path.join(os.getcwd(),'..','Anomaly_Detection_splits/Normal_Test_clean.txt')
    normal_files = open(normal_file_path,'r')
    normal_videos = normal_files.readlines()
    normal_videos = [line.split('/')[1].strip("\n") for line in normal_videos]

    anomaly_file_path = os.path.join(os.getcwd(),'..','Anomaly_Detection_splits/Anomaly_Test.txt')
    anomaly_files = open(anomaly_file_path,'r')
    anomaly_videos = anomaly_files.readlines()
    anomaly_videos = [line.split('/')[1].strip("\n") for line in anomaly_videos]

    all_videos = normal_videos + anomaly_videos
    #print(all_videos)

    # load validation annotations txt file
    file_path = os.path.join(os.getcwd(), '..', 'Anomaly_Detection_splits/Temporal_annotations_for_vald.txt')
    file_obj = open(file_path, 'r')
    all_lines = file_obj.readlines()
    new_list = []
    for lines in all_lines:
        #print(lines.split(' ')[0])
        if lines.split(' ')[0] in all_videos:
            new_list.append(lines)
        else:
            print(lines.split(' ')[0], 'is greater than 20 minutes, so ignored')

    filename = os.path.join(os.getcwd(), '..', 'Anomaly_Detection_splits/Temporal_annotations_for_vald_clean.txt')
    if not os.path.exists(filename):
        with open(filename, 'w') as f:
            for item in new_list:
                f.write("%s" % item)
