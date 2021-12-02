import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim
import os
from numpy import array, zeros, full, argmin, inf, ndim
from math import isinf
from scipy.spatial.distance import cdist
from numba import jit

from utils import parse_args,model_dict
from dataset import SetDataManager
from meta_template import BaseNet

def train(data_loader_list, model, optimization, start_epoch, stop_epoch, params):    
    [base_loader, val_loader, test_loader] = data_loader_list
    if optimization == 'Adam':
        lr = params.lr
        print('lr=', lr)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
       raise ValueError('Unknown optimization, please define by yourself')

    max_acc = 0    
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(start_epoch,stop_epoch):
        model.train()
        avg_loss=0
        for i, (x,_ ) in enumerate(base_loader):
            # print(x.shape) # [n_way, support+query, T, C, H, W] [5,2,8,3,224,224]
            x = x.cuda()
            nway, sq, t, c, h, w = x.shape
            x = x.reshape(nway*sq*t, c, h, w) # 80 images
            logits = model(x) # 80 x 2048
            logits = logits.reshape(nway,sq,t,logits.shape[-1]) # 5x2x8x2048
            if isinstance(model, nn.DataParallel):
                n_support = model.module.n_support
            else:
                n_support = model.n_support
            
            n_query = sq - n_support
            z_support   = logits[:, :n_support].reshape(nway * n_support, t, -1)
            z_query     = logits[:, n_support:].reshape(nway * n_query, t, -1)
            scores = torch.zeros((nway*n_query, nway*n_support),requires_grad=True).cuda()
            for qi in range(nway*n_query): # query i
                for si in range(nway*n_support): # support i
                    x = z_support[si].cpu().detach().numpy()
                    y = z_query[qi].cpu().detach().numpy()
                    _, path = dtw(x,y)
                    #print(path)
                    start = np.where(path[0]==1)[0][0]
                    x_index = path[0][start:start+t]-1 # subtract the extra coordinates
                    y_index = path[1][start:start+t]
                    scores[qi][si] = F.cosine_similarity(z_support[si][x_index], z_query[qi][y_index]).sum()
                    
                    _, path = dtw(y,x)
                    start = np.where(path[0]==1)[0][0]
                    x_index = path[0][start:start+t]-1 # subtract the extra coordinates
                    y_index = path[1][start:start+t]
                    scores[qi][si] += F.cosine_similarity(z_support[si][y_index], z_query[qi][x_index]).sum()

            scores = scores.reshape(nway*n_query,nway,n_support).mean(2)
            
            y_query = torch.from_numpy(np.repeat(range( nway ), n_query ))
            y_query = Variable(y_query.cuda())
            
            loss = loss_fn(scores,y_query)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            avg_loss = avg_loss+loss.item()
        
        print('Epoch {:d} | Loss {:f} | '.format(epoch, avg_loss/float(i+1)),end="")
        model.eval()

        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)
        acc_all = []
        iter_num = len(val_loader)
        with torch.no_grad():
            for i, (x,_) in enumerate(val_loader):
                x = x.cuda()
                nway, sq, t, c, h, w = x.shape
                x = x.reshape(nway*sq*t, c, h, w) # 80 images
                logits = model(x) # 80 x 2048
                logits = logits.reshape(nway,sq,t,logits.shape[-1]) # 5x2x8x2048
                if isinstance(model, nn.DataParallel):
                    n_support = model.module.n_support
                else:
                    n_support = model.n_support
                
                n_query = sq - n_support
                z_support   = logits[:, :n_support].reshape(nway * n_support, t, -1)
                z_query     = logits[:, n_support:].reshape(nway * n_query, t, -1)
                scores = torch.zeros((nway*n_query, nway*n_support),requires_grad=True).cuda()
                
                for qi in range(nway*n_query): # query i
                    for si in range(nway*n_support): # nway i
                        x = z_support[si].cpu().detach().numpy()
                        y = z_query[qi].cpu().detach().numpy()
                        _, path = dtw(x,y)
                        start = np.where(path[0]==1)[0][0]
                        x_index = path[0][start:start+t]-1 # subtract the extra coordinates
                        y_index = path[1][start:start+t]
                        scores[qi][si] = F.cosine_similarity(z_support[si][x_index], z_query[qi][y_index]).sum()

                        _, path = dtw(y,x)
                        start = np.where(path[0]==1)[0][0]
                        x_index = path[0][start:start+t]-1 # subtract the extra coordinates
                        y_index = path[1][start:start+t]
                        scores[qi][si] += F.cosine_similarity(z_support[si][y_index], z_query[qi][x_index]).sum()
                
                scores = scores.reshape(nway*n_query,nway,n_support).mean(2)
                y_query = np.repeat(range( nway ), n_query )
                topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
                topk_ind = topk_labels.cpu().numpy()
                top1_correct = np.sum(topk_ind[:,0] == y_query)
                correct_this, count_this = float(top1_correct), len(y_query)
                acc_all.append(correct_this/ count_this*100  )

        acc_all  = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std  = np.std(acc_all)
        print('%d Val Acc = %4.2f%% +- %4.2f%%' %(iter_num,  acc_mean, 1.96* acc_std/np.sqrt(iter_num)))

        acc = acc_mean
        if acc > max_acc:
            print("best model! save...")
            max_acc = acc
            outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
            torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)
        if epoch % params.save_freq == 0:
            outfile = os.path.join(params.checkpoint_dir, 'epoch{}'.format(epoch))
            torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)
        
    return model

def test(test_loader, model, params):
    model.eval()
    acc_all = []
    iter_num = len(test_loader)
    with torch.no_grad():
        for i, (x,_) in enumerate(test_loader):
            x = x.cuda()
            nway, sq, t, c, h, w = x.shape
            x = x.reshape(nway*sq*t, c, h, w) # 80 images
            logits = model(x) # 80 x 2048
            logits = logits.reshape(nway,sq,t,logits.shape[-1]) # 5x2x8x2048
            if isinstance(model, nn.DataParallel):
                n_support = model.module.n_support
            else:
                n_support = model.n_support
            
            n_query = sq - n_support
            z_support   = logits[:, :n_support].reshape(nway * n_support, t, -1)
            z_query     = logits[:, n_support:].reshape(nway * n_query, t, -1)
            scores = torch.zeros((nway*n_query, nway*n_support),requires_grad=True).cuda()
            
            for qi in range(nway*n_query): # query i
                for si in range(nway*n_support): # nway i
                    x = z_support[si].cpu().detach().numpy()
                    y = z_query[qi].cpu().detach().numpy()
                    _, path = dtw(x,y)
                    start = np.where(path[0]==1)[0][0]
                    x_index = path[0][start:start+t]-1 # subtract the extra coordinates
                    y_index = path[1][start:start+t]
                    scores[qi][si] = F.cosine_similarity(z_support[si][x_index], z_query[qi][y_index]).sum()

                    _, path = dtw(y,x)
                    start = np.where(path[0]==1)[0][0]
                    x_index = path[0][start:start+t]-1 # subtract the extra coordinates
                    y_index = path[1][start:start+t]
                    scores[qi][si] += F.cosine_similarity(z_support[si][y_index], z_query[qi][x_index]).sum()
            
            scores = scores.reshape(nway*n_query,nway,n_support).mean(2)

            y_query = np.repeat(range( nway ), n_query )
            topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
            topk_ind = topk_labels.cpu().numpy()
            top1_correct = np.sum(topk_ind[:,0] == y_query)
            correct_this, count_this = float(top1_correct), len(y_query)
            acc_all.append(correct_this/ count_this*100  )
    acc_all  = np.asarray(acc_all)
    acc_mean = np.mean(acc_all)
    acc_std  = np.std(acc_all)
    print('%d Test Acc = %4.2f%% +- %4.2f%%' %(iter_num,  acc_mean, 1.96* acc_std/np.sqrt(iter_num)))

def euclidean_dist( x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


def dtw(x, y):

    r, c = len(x), len(y)
    assert r == c
    r = r + 2
    D0 = np.zeros((r + 1, c + 1))
    D0[0, 1:] = np.inf
    D0[1:, 0] = np.inf
    D1 = D0[1:, 1:]  # view
    
    # for i in range(1,r-1): # non zero
        # for j in range(c):
            # D1[i, j] = dist(x[i-1], y[j])
    
    n = x.shape[0]
    m = y.shape[0]
    d = x.shape[1]
    assert n == m
    assert d == y.shape[1]

    if False: #Euclidean
        x = np.broadcast_to(np.expand_dims(x,1),(n,m,d)) 
        y = np.broadcast_to(np.expand_dims(y,0),(n,m,d))

        D1[1:r-1,0:c] = np.power(x - y, 2).sum(2)
    
    else: #cosine
        D1[1:r-1,0:c] = cdist(x,y,'cosine')
    # print(D1)

    D0,D1 = dtw_loop(r,c,D0,D1)
    
    if len(x) == 1:
        path = np.zeros(len(y)), range(len(y))
    elif len(y) == 1:
        path = range(len(x)), np.zeros(len(x))
    else:
        path = _traceback(D0)
    # return D1[-1, -1], C, D1, path
    return D1[-1, -1], path

@jit(nopython=True)
def dtw_loop(r,c,D0,D1):
    for i in range(r): # [0,T+1]
        for j in range(c): # [0,T-1]
            # min_list = [D0[i, j]]
            i_k = min(i + 1, r)
            j_k = min(j + 1, c)
            if i == 0 or i == r - 1: # first and last 
                # min_list += [D0[i_k, j] * s, D0[i, j_k] * s]
                min_list = [D0[i, j],D0[i_k, j],D0[i, j_k]]
            else:
                # min_list += [D0[i, j_k] * s]
                min_list = [D0[i, j], D0[i, j_k]]
            # D1[i, j] += min(min_list) # smooth version
            # print(min_list)
            
            D1[i, j] += min(min_list)
    return D0,D1

def _traceback(D):
    i, j = np.array(D.shape) - 2
    p, q = [i], [j]
    while (i > 0) or (j > 0):
        tb = np.argmin((D[i, j], D[i, j + 1], D[i + 1, j]))
        if tb == 0:
            i -= 1
            j -= 1
        elif tb == 1:
            i -= 1
        else:  # (tb == 2):
            j -= 1
        p.insert(0, i)
        q.insert(0, j)
    return np.array(p), np.array(q)

if __name__=='__main__':
    np.random.seed(10)
    params = parse_args('train')
    if params.dataset == 'kinetics':
        base_file = '/home/zzx/workspace2/data/kinetics_frames_jpg/annotations/train.txt'
        val_file ='/home/zzx/workspace2/data/kinetics_frames_jpg/annotations/val.txt'
        test_file = '/home/zzx/workspace2/data/kinetics_frames_jpg/annotations/test.txt'
    elif params.dataset == 'somethingotam':
        base_file = '/home/zzx/workspace2/data/smsm_otam_frames/annotations/train.txt'
        val_file = '/home/zzx/workspace2/data/smsm_otam_frames/annotations/val.txt'
        test_file = '/home/zzx/workspace2/data/smsm_otam_frames/annotations/test.txt'
    elif params.dataset == 'kineticsnew':
        base_file = '/home/zzx/workspace2/data/kinetics_newbenchmark_frames/annotations/train.txt'
        val_file ='/home/zzx/workspace2/data/kinetics_newbenchmark_frames/annotations/val.txt'
        test_file = '/home/zzx/workspace2/data/kinetics_newbenchmark_frames/annotations/test.txt'
    else:
        raise ValueError('Unknown dataset')
    
    image_size = 224
    optimization = 'Adam'

    if params.stop_epoch == -1: 
        params.stop_epoch = 400
     
    params.method = 'otam'
    if params.method in ['otam']:
        n_query = max(1, int(params.n_query* params.test_n_way/params.train_n_way)) #if test_n_way is smaller than train_n_way, reduce n_query to keep batch size small
 
        train_few_shot_params    = dict(n_way = params.train_n_way, n_support = params.n_shot) 
        base_datamgr            = SetDataManager(image_size, n_query = n_query,  num_segments = params.num_segments, **train_few_shot_params)
        base_loader             = base_datamgr.get_data_loader( base_file , aug = params.train_aug )
         
        test_few_shot_params     = dict(n_way = params.test_n_way, n_support = params.n_shot) 
        val_datamgr             = SetDataManager(image_size, n_query = n_query,  num_segments = params.num_segments, n_eposide = params.eval_episode, **test_few_shot_params)
        val_loader              = val_datamgr.get_data_loader( val_file, aug = False) 

        test_datamgr             = SetDataManager(image_size, n_query = n_query,  num_segments = params.num_segments, n_eposide = params.test_episode, **test_few_shot_params)
        test_loader              = test_datamgr.get_data_loader( test_file, aug = False) 
        #a batch for SetDataManager: a [n_way, n_support + n_query, dim, w, h] tensor        

        model = BaseNet( model_dict[params.model], **train_few_shot_params )
    else:
       raise ValueError('Unknown method')

    model = model.cuda()
    if params.test_model:
        checkpoint = torch.load(params.checkpoint,map_location=lambda storage, loc: storage.cuda(0))
        checkpoint = checkpoint['state']
        base_dict = {}
        for k, v in list(checkpoint.items()):
            if k.startswith('module'):
                base_dict['.'.join(k.split('.')[1:])] = v
            else:
                base_dict[k] = v
        model.load_state_dict(base_dict)
        test(test_loader, model, params)
    else:
        model = nn.DataParallel(model)
        params.checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(params.work_dir, params.dataset, params.model, params.method)
        if params.train_aug:
            params.checkpoint_dir += '_aug'
        params.checkpoint_dir += '_%dway_%dshot' %( params.train_n_way, params.n_shot)

        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)

        start_epoch = params.start_epoch
        stop_epoch = params.stop_epoch
        data_loader_list = [base_loader, val_loader, test_loader]
        model = train(data_loader_list,  model, optimization, start_epoch, stop_epoch, params)
