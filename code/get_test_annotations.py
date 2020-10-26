import os

def get_test_annotations(f_path):
    """
    Input:
    'f_path' : location of txt file that contains segment wise annotations for test dataset
    Output:
    'test_dset_list' : a list where
    each entry of list is dictionary with keys:
    'file_path' : location of test file,
    'category' : category of test file,
    'num_frames' : total number of frames in video file
    'anomaly_frames' : np array of segments in which anomaly is present
    """
    # Read test annotations file
    f_obj = open(f_path, 'r')
    alllines = f_obj.readlines()
    test_dset_list = []
    video_dict = {}

    for line in alllines:
        file_name = line.split(' ')[0]
        # print(file_name)
        category_name = line.split(' ')[2]
        # print(category_name)
        video_path = '/content/drive/Shared drives/EECS 545 - ML Project/data/Video_files/'
        if category_name in ["Arson", "Arrest", "Abuse", "Assault"]:
            folder_name = 'Anomaly-Videos-Part-1/' + category_name + '/'
        elif category_name in ["Explosion", "Fighting", "Burglary"]:
            folder_name = 'Anomaly-Videos-Part-2/' + category_name + '/'
        elif category_name in ["Shooting", "RoadAccidents", "Robbery"]:
            folder_name = 'Anomaly-Videos-Part-3/' + category_name + '/'
        elif category_name in ["Shoplifting", "Stealing", "Vandalism"]:
            folder_name = 'Anomaly-Videos-Part-4/' + category_name + '/'
        elif category_name == "Normal":
            folder_name = 'Testing_Normal_Videos_Anomaly/'
        else:
            raise Exception('Category :', category_name, ' name not in defined categories')

        file_path = video_path + folder_name + file_name
        # print(file_path)
        # num_frames = torchvision.io.read_video(file_path)[0].shape[0]
        num_frames = count_frames(file_path)
        new = video_dict.copy()
        new['file_path'] = file_path
        new['num_frames'] = num_frames
        new['category'] = category_name

        # print(num_frames)
        start_frame1 = int(line.split(' ')[4])
        end_frame1 = int(line.split(' ')[6])
        if end_frame1 >= num_frames:
            end_frame1 = num_frames - 1
            if end_frame1 > num_frames:
                print('End frame = ', end_frame1, 'Num of frames = ', num_frames)
                raise Exception('end frame cannot be larger than number of frames')
        # print(start_frame1, end_frame1)
        if start_frame1 != -1:
            anomaly_frame1 = np.floor((start_frame1 / num_frames) * 32).astype(int)
            anomaly_frame2 = np.floor((end_frame1 / num_frames) * 32).astype(int)
            anomaly_frame_idx1 = np.arange(anomaly_frame1, anomaly_frame2 + 1, 1, dtype=int)
        else:
            anomaly_frame_idx1 = np.array([])
        # print(anomaly_frame_idx1)

        start_frame2 = int(line.split(' ')[8])
        end_frame2 = int(line.split(' ')[10])
        if end_frame2 >= num_frames:
            end_frame2 = num_frames - 1
            if end_frame2 > num_frames:
                print('End frame = ', end_frame2, 'Num of frames = ', num_frames)
                raise Exception('end frame cannot be larger than number of frames')
        # print(start_frame2, end_frame2)
        if start_frame2 != -1:
            anomaly_frame1 = np.floor((start_frame2 / num_frames) * 32).astype(int)
            anomaly_frame2 = np.floor((end_frame2 / num_frames) * 32).astype(int)
            anomaly_frame_idx2 = np.arange(anomaly_frame1, anomaly_frame2 + 1, 1, dtype=int)
        else:
            anomaly_frame_idx2 = np.array([])
        # print(anomaly_frame_idx2)
        anomaly_frames = np.concatenate((anomaly_frame_idx1, anomaly_frame_idx2)).astype(int)
        # print(anomaly_frames.dtype)
        if 32 in anomaly_frames:
            raise Exception('anomaly frame number cannot be 32')

        new['anomaly_frames'] = anomaly_frames
        print('Test file : ', file_name, 'category : ', category_name, 'anomaly frames = ', anomaly_frames)
        test_dset_list.append(new)
    return test_dset_list

if __name__ == '__main__':
    f_path = os.path.join(os.getcwd(),"..",'Anomaly_Detection_splits','Temporal_Anomaly_Annotation_for_Testing_Videos.txt')
    test_annotations = get_test_annotations(f_path)
    print('Sample entry in the list : ', test_annotations[16])
    # save test_annotations to saved_data folder
    write_path = os.path.join(os.getcwd(),"..",'saved_data','test_annotations.pt')
    if ~os.exists(write_path):
        torch.save(test_annotations,write_path)