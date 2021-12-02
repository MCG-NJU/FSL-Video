import os
import yaml
import backbone
import torch
import torch.nn as nn
import numpy as np
from save_features import save_features,get_best_file
from test import BaselineFinetune, model_dict, feature_evaluation, init_loader
from tsn_loader import SimpleDataManager
from tqdm import tqdm
import sys
sys.path.append('/home/zzx/workspace/code/temporal-adaptive-module/')
from ops.models import TSN

class Identity(torch.nn.Module):
    def forward(self, input):
        return input

if __name__ == '__main__':
    config = '/home/zzx/workspace/code/video_FSL/bmvc21/config/test_baseline.yaml'
    
    with open(config, 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    
    assert params['method'] in ['support','baseline']
    iter_num = params['iter_num']
    few_shot_params = dict(n_way = params['test_n_way'] , n_support = params['n_shot'])
    split = params['split']
    batch_size = params['batch_size']
    num_segments = params['num_segments']
    file = os.path.join(params['data_dir'],'{}.txt'.format(split))
    print(file)
    if params['method'] == 'support':
        dropout = 0.5
    elif params['method'] == 'baseline':
        dropout = 0.0
    
    checkpoint_dir = params['checkpoint']
    split = params['split']
    split_str = split
    novel_file = os.path.join( checkpoint_dir.replace("checkpoints","features"), split_str +".hdf5")
    outfile = novel_file
    
    print_once = True
    best_acc_mean = 0
    for model_file in os.listdir(checkpoint_dir): 
        if os.path.splitext(model_file)[-1] != '.tar':
            continue

        path = os.path.join(checkpoint_dir, model_file)
        print ('\n',path) 
        
        dirname = os.path.dirname(outfile)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)

        # load model
        base_class = params['base_classes']
        model = TSN(params['base_classes'],8,'RGB','resnet50',tam=params['tam'],print_spec=print_once,dropout=dropout)
        model = model.cuda()
        print_once = False

        image_size = 224
        datamgr    = SimpleDataManager(image_size, batch_size = batch_size, num_segments=num_segments)
        data_loader     = datamgr.get_data_loader( data_file = file , aug = False )

        checkpoint = torch.load(path,map_location=lambda storage, loc: storage.cuda(0))
        for key,value in checkpoint.items():
            if key in ['epoch','arch'] :
                print(key,value)
        checkpoint = checkpoint['state_dict']
        base_dict = {}
        for k, v in list(checkpoint.items()):
            if k.startswith('module'):
                base_dict['.'.join(k.split('.')[1:])] = v
            else:
                base_dict[k] = v
        
        
        model.load_state_dict(base_dict)
        
        if params['append_classifier']:
            final_feat_dim = 64
        else:
            if params['method'] == 'support':
                model.new_fc = Identity()
            elif params['method'] == 'baseline':
                model.base_model.fc = Identity() # remove 2048 x 64
            final_feat_dim = 2048
        # save features
        model.eval() 
        with torch.no_grad():  
            save_features(model, data_loader, outfile, False, tam=True)
        del model
        torch.cuda.empty_cache()

        # test
        max_acc_all = []
        acc_all = []
        cumulative_acc_all = []
        cl_data_file = init_loader(novel_file)

        if params['method'] == 'baseline':
            model = BaselineFinetune(model_dict[params['model']], final_feat_dim=final_feat_dim, **few_shot_params)
        elif params['method'] == 'support':
            model = BaselineFinetune(model_dict[params['model']], base_class=base_class, loss_type = 'support', final_feat_dim=final_feat_dim, **few_shot_params)
        
        model = model.cuda()

        for i in range(iter_num):
            acc = feature_evaluation(cl_data_file, model, n_query = params['n_query'], adaptation = True, **few_shot_params)
            acc_all.append(acc)
            
        
        acc_all  = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std  = np.std(acc_all)
        print('%d Test Acc = %4.2f%% +- %4.2f%%' %(iter_num, acc_mean, 1.96* acc_std/np.sqrt(iter_num)))
        
    
        