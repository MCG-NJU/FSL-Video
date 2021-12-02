import os
import numpy as np
import torch
from torch.autograd import Variable
import glob
import h5py
import yaml
import backbone

from utils import model_dict
from tsn_loader import VideoDataset 
from tqdm import tqdm


def get_resume_file(checkpoint_dir,test_epoch=None):
    filelist = glob.glob(os.path.join(checkpoint_dir, '*.tar'))
    if len(filelist) == 0:
        return None

    filelist =  [ x  for x in filelist if os.path.basename(x) != 'best_model.tar' ]
    epochs = np.array([int(os.path.splitext(os.path.basename(x))[0]) for x in filelist])
    max_epoch = np.max(epochs)
    # max_epoch = 100
    if test_epoch is not None:
        max_epoch = test_epoch
    resume_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(max_epoch))
    return resume_file
    
def get_best_file(checkpoint_dir,test_epoch=None):    
    best_file = os.path.join(checkpoint_dir, 'best_model.tar')
    if os.path.isfile(best_file):
        return best_file
    else:
        return get_resume_file(checkpoint_dir,test_epoch)

def save_features(model, data_loader, outfile, temporal_aug=False, tam=False):
    f = h5py.File(outfile, 'w')
    max_count = len(data_loader)*data_loader.batch_size
    all_labels = f.create_dataset('all_labels',(max_count,), dtype='i')
    all_feats=None
    count=0
    model.eval()
    
    print_once = True
    for i, (x,y) in tqdm(enumerate(data_loader)):
        x = x.cuda()
        x_var = Variable(x)
        if temporal_aug:
            feats = None
            for seg in range(x_var.shape[1]):
                feat_seg = model(x_var[:,seg,:,:,:])  # N x C
                if feats is None:
                    N,C = feat_seg.shape
                    feats = torch.zeros((N,x_var.shape[1],C)) # N x T x C
                feats[:,seg,:] = feat_seg
        elif tam:
            feats = model(x_var)
        else:
            feats = 0
            for seg in range(x_var.shape[1]):
                feats += model(x_var[:,seg,:,:,:]) 
            feats = feats / x_var.shape[1]
        # feats = model(x_var)
        if all_feats is None:
            all_feats = f.create_dataset('all_feats', [max_count] + list( feats.size()[1:]) , dtype='f')
        if print_once:
            print_once= False
            print(feats.shape)
        all_feats[count:count+feats.size(0)] = feats.data.cpu().numpy()
        all_labels[count:count+feats.size(0)] = y.cpu().numpy()
        count = count + feats.size(0)


    count_var = f.create_dataset('count', (1,), dtype='i')
    count_var[0] = count

    f.close()

