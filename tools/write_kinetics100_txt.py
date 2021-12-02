import numpy as np
from os import listdir
from os.path import isfile, isdir, join
import os
import json
import random
import glob


# '/home/zzx/workspace/data/kinetics_100'
# frame_path = '/home/zzx/workspace2/data/smsm_otam_frames/frames'
frame_path = '/home/zzx/workspace2/data/kinetics_newbenchmark_frames/frames'
# data_path = '/home/zzx/workspace/data/kinetics_100'
# data_path = '/home/zzx/workspace2/data/smsm_otam_videos'
data_path = '/home/zzx/workspace2/data/kinetics_newbenchmark_videos'

# savedir = '/home/zzx/workspace2/data'
# dataset_list = ['base','val','novel']
dataset_list = ['base']

folder_list = [f for f in listdir(data_path) if isdir(join(data_path, f))]
folder_list.sort()
label_dict = dict(zip(folder_list,range(0,len(folder_list))))

classfile_list_all = []

for i, folder in enumerate(folder_list):
    folder_path = join(data_path, folder)
    classfile_list_all.append( [ join(folder_path, cf) for cf in listdir(folder_path) if (isfile(join(folder_path,cf)) and cf[0] != '.')])
    random.shuffle(classfile_list_all[i])


for dataset in dataset_list:
    file_list = []
    label_list = []
    for i, classfile_list in enumerate(classfile_list_all):
        # train: 001-064  val:065-076  test:077-100
        if 'base' in dataset:
            # if (i%2 == 0):
            if i <= 63:
                f = open('/home/zzx/workspace2/data/tools/train.txt',mode='a')
                for video_path in classfile_list:
                    path = video_path
                    name = os.path.basename(path)
                    name = os.path.splitext(name)[0]
                    name = os.path.join(frame_path,name)
                    frames = len(glob.glob(pathname=os.path.join(name,'*.*')))
                    label = i
                    f.write('{} {} {}\n'.format(name,frames,label))
                # file_list = file_list + classfile_list
                # label_list = label_list + np.repeat(i, len(classfile_list)).tolist()
        if 'val' in dataset:
            # if (i%4 == 1):
            if i > 63 and i <=75:
                f = open('/home/zzx/workspace2/data/tools/val.txt',mode='a')
                for video_path in classfile_list:
                    path = video_path
                    name = os.path.basename(path)
                    name = os.path.splitext(name)[0]
                    name = os.path.join(frame_path,name)
                    frames = len(glob.glob(pathname=os.path.join(name,'*.*')))
                    label = i
                    f.write('{} {} {}\n'.format(name,frames,label))
        if 'novel' in dataset:
            # if (i%4 == 3):
            if i > 75:
                f = open('/home/zzx/workspace2/data/tools/test.txt',mode='a')
                for video_path in classfile_list:
                    path = video_path
                    name = os.path.basename(path)
                    name = os.path.splitext(name)[0]
                    name = os.path.join(frame_path,name)
                    frames = len(glob.glob(pathname=os.path.join(name,'*.*')))
                    label = i
                    f.write('{} {} {}\n'.format(name,frames,label))

    '''
    fo = open(savedir + dataset + ".json", "w")
    fo.write('{"label_names": [')
    fo.writelines(['"%s",' % item  for item in folder_list])
    fo.seek(0, os.SEEK_END) 
    fo.seek(fo.tell()-1, os.SEEK_SET)
    fo.write('],')

    fo.write('"image_names": [')
    fo.writelines(['"%s",' % item  for item in file_list])
    fo.seek(0, os.SEEK_END) 
    fo.seek(fo.tell()-1, os.SEEK_SET)
    fo.write('],')

    fo.write('"image_labels": [')
    fo.writelines(['%d,' % item  for item in label_list])
    fo.seek(0, os.SEEK_END) 
    fo.seek(fo.tell()-1, os.SEEK_SET)
    fo.write(']}')

    fo.close()
    '''
    print("%s -OK" %dataset)