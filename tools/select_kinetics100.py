import os
import shutil
from os.path import isfile, isdir, join

def print_log(str, log_dir, print_to_screen=True, save_log=True):
    if print_to_screen:
        print(str)
    if save_log:
        with open('{}/log.txt'.format(log_dir), 'a') as f:
        # with open(log_dir, 'a') as f:
            print(str, file=f)


train_all_path = 'D:/data/kinetics/kinetics400/Kinetics_trimmed_processed_train'
val_all_path = 'D:/data/kinetics/kinetics400/Kinetics_trimmed_processed_val'

video_all_path = '/home/zzx/workspace2/data/something_v2/20bn-something-something-v2'
# train_list_100_path = 'D:/data/kinetics/select_kinetics/train.list'
# train_list_100_path = 'D:/data/kinetics/select_kinetics/val.list'
# train_list_100_path = '/home/zzx/workspace2/data/split/smsm-100/test.list'
train_list_100_path = '/home/zzx/workspace2/data/split/smsm-otam/testlist07.txt'

output_path = '/home/zzx/workspace2/data/smsm_otam_videos'
log_dir = '/home/zzx/workspace2/data'
# if os.path.exists(output_path):
    # os.remove(output_path)

file = open(train_list_100_path, 'r')
start_idx = 77 # train:1  val:65  test:77
class_dict = {}
loop_num = 1
kinetics = False
for line in file:
    # print(line)
    rest_info = line.split('/')[1]
    if kinetics: # kinetics
        id = rest_info[:11] # video name
    else: # sth_v2
        id = rest_info[:-1]
    line = line.split('/')[0] # label name
    
    # make class folder
    if not line in class_dict:
        folder_name = str(start_idx).rjust(3,'0') + '.' + line.replace(' ','_')
        folder_path = join(output_path,folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        class_dict[line] = start_idx
        start_idx += 1
    
    if kinetics:
        video_path = train_all_path + '/'+ id + '.mp4'
    else:
        video_path = video_all_path + '/'+ id + '.webm'
    if not os.path.exists(video_path):
        video_path = val_all_path + '/'+ id + '.mp4'
        if not os.path.exists(video_path): # missing!
            log = '{}/{}'.format(line,id)
            print_log(log,log_dir)
            continue

    folder_name = str(class_dict[line]).rjust(3,'0') + '.' + line.replace(' ','_')
    folder_path = output_path + '/' + folder_name
    # os.system('cp video_path kinetics_train_midframe/') # linux
    cmd = 'cp {} {}'.format(video_path,folder_path)
    os.system(cmd)
    # cmd = 'copy {} {}'.format(video_path,folder_path)
    print(cmd)
    shutil.copy(video_path,folder_path)
    


    loop_num +=1
    if loop_num>=5:
        # break
        continue
        # break
    # print(rest_info)
    # print(id)
    # print(line)
    
'''
for i in range(1,30):
    print (str(i).rjust(3,'0'))
'''
