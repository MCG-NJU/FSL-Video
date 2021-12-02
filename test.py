import os
import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from abc import abstractmethod
import yaml
import argparse
import h5py
import random
from tqdm import tqdm
from multiprocessing import Pool
from utils import model_dict

class SimpleHDF5Dataset:
    def __init__(self, file_handle = None):
        if file_handle == None:
            self.f = ''
            self.all_feats_dset = []
            self.all_labels = []
            self.total = 0 
        else:
            self.f = file_handle
            self.all_feats_dset = self.f['all_feats'][...]
            self.all_labels = self.f['all_labels'][...]
            self.total = self.f['count'][0]
           # print('here')
    def __getitem__(self, i):
        return torch.Tensor(self.all_feats_dset[i,:]), int(self.all_labels[i])

    def __len__(self):
        return self.total

def init_loader(filename):
    with h5py.File(filename, 'r') as f:
        fileset = SimpleHDF5Dataset(f)

    #labels = [ l for l  in fileset.all_labels if l != 0]
    feats = fileset.all_feats_dset
    labels = fileset.all_labels
    while np.sum(feats[-1]) == 0:
        feats  = np.delete(feats,-1,axis = 0)
        labels = np.delete(labels,-1,axis = 0)
        
    class_list = np.unique(np.array(labels)).tolist() 
    inds = range(len(labels))

    cl_data_file = {}
    for cl in class_list:
        cl_data_file[cl] = []
    for ind in inds:
        cl_data_file[labels[ind]].append( feats[ind])

    return cl_data_file

class MetaTemplate(nn.Module):
    def __init__(self, model_func, n_way, n_support, base_class = None, loss_type = None, change_way = True):
        super(MetaTemplate, self).__init__()
        self.n_way      = n_way
        self.n_support  = n_support
        self.n_query    = -1 #(change depends on input)
        self.change_way = change_way  #some methods allow different_way classification during training and test

    @abstractmethod
    def set_forward(self,x,is_feature):
        pass

    @abstractmethod
    def set_forward_loss(self, x):
        pass

    def forward(self,x):
        pass

    def parse_feature(self,x,is_feature):
        x    = Variable(x.cuda())
        if is_feature:
            z_all = x
        else:
            x           = x.contiguous().view( self.n_way * (self.n_support + self.n_query), *x.size()[2:]) 
            z_all       = self.feature.forward(x)
            z_all       = z_all.view( self.n_way, self.n_support + self.n_query, -1)
        z_support   = z_all[:, :self.n_support]
        z_query     = z_all[:, self.n_support:]

        return z_support, z_query

    def correct(self, x):       
        scores = self.set_forward(x)
        y_query = np.repeat(range( self.n_way ), self.n_query )

        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:,0] == y_query)
        return float(top1_correct), len(y_query)

    def train_loop(self, epoch, train_loader, optimizer ):
        print_freq = 10

        avg_loss=0
        for i, (x,_ ) in enumerate(train_loader):
            self.n_query = x.size(1) - self.n_support           
            if self.change_way:
                self.n_way  = x.size(0)
            optimizer.zero_grad()
            loss = self.set_forward_loss( x )
            loss.backward()
            optimizer.step()
            avg_loss = avg_loss+loss.data[0]

            if i % print_freq==0:
                #print(optimizer.state_dict()['param_groups'][0]['lr'])
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader), avg_loss/float(i+1)))

    def test_loop(self, test_loader, record = None):
        correct =0
        count = 0
        acc_all = []
        
        iter_num = len(test_loader) 
        for i, (x,_) in enumerate(test_loader):
            self.n_query = x.size(1) - self.n_support
            if self.change_way:
                self.n_way  = x.size(0)
            correct_this, count_this = self.correct(x)
            acc_all.append(correct_this/ count_this*100  )

        acc_all  = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std  = np.std(acc_all)
        print('%d Test Acc = %4.2f%% +- %4.2f%%' %(iter_num,  acc_mean, 1.96* acc_std/np.sqrt(iter_num)))

        return acc_mean

    def set_forward_adaptation(self, x, is_feature = True): #further adaptation, default is fixing feature and train a new softmax clasifier
        assert is_feature == True, 'Feature is fixed in further adaptation'
        z_support, z_query  = self.parse_feature(x,is_feature)

        z_support   = z_support.contiguous().view(self.n_way* self.n_support, -1 )
        z_query     = z_query.contiguous().view(self.n_way* self.n_query, -1 )

        y_support = torch.from_numpy(np.repeat(range( self.n_way ), self.n_support ))
        y_support = Variable(y_support.cuda())

        linear_clf = nn.Linear(self.feat_dim, self.n_way)
        linear_clf = linear_clf.cuda()

        set_optimizer = torch.optim.SGD(linear_clf.parameters(), lr = 0.01, momentum=0.9, dampening=0.9, weight_decay=0.001)

        loss_function = nn.CrossEntropyLoss()
        loss_function = loss_function.cuda()
        
        batch_size = 4
        support_size = self.n_way* self.n_support
        for epoch in range(100):
            rand_id = np.random.permutation(support_size)
            for i in range(0, support_size , batch_size):
                set_optimizer.zero_grad()
                selected_id = torch.from_numpy( rand_id[i: min(i+batch_size, support_size) ]).cuda()
                z_batch = z_support[selected_id]
                y_batch = y_support[selected_id] 
                scores = linear_clf(z_batch)
                loss = loss_function(scores,y_batch)
                loss.backward()
                set_optimizer.step()

        scores = linear_clf(z_query)
        return scores

class BaselineFinetune(MetaTemplate):
    def __init__(self, model_func, n_way, n_support, base_class = None, loss_type = "softmax", final_feat_dim=2048):
        super(BaselineFinetune, self).__init__( model_func,  n_way, n_support, base_class, loss_type)
        self.loss_type = loss_type
        self.feat_dim = final_feat_dim

    def set_forward(self,x,is_feature = True):
        return self.set_forward_adaptation(x,is_feature)
 
    def set_forward_adaptation(self,x,is_feature = True,temporal_aug=False):
        assert is_feature == True
        z_support, z_query  = self.parse_feature(x,is_feature)
        '''
        z_support   = z_support.contiguous().view(self.n_way* self.n_support, -1 )
        z_query     = z_query.contiguous().view(self.n_way* self.n_query, -1 )
        '''
        if temporal_aug:
            T = x.shape[-2]
            z_support   = z_support.contiguous().view(self.n_way* self.n_support* T, -1 )
            z_query     = z_query.contiguous().view(self.n_way* self.n_query* T, -1 )
            y_support = torch.from_numpy(np.repeat(range( self.n_way ), self.n_support* T))
        else:
            z_support   = z_support.contiguous().view(self.n_way* self.n_support, -1 )
            z_query     = z_query.contiguous().view(self.n_way* self.n_query, -1 )
            y_support = torch.from_numpy(np.repeat(range( self.n_way ), self.n_support ))
        y_support = Variable(y_support.cuda())

        if self.loss_type == 'softmax':
            linear_clf = nn.Linear(self.feat_dim, self.n_way)
        elif self.loss_type == 'support': 
            # linear clf
            linear_clf = nn.Linear(self.feat_dim, self.n_way)
            # support based initialization
            # print(z_support.shape)
            # assert z_support.shape[0] == self.n_way
            if z_support.shape[0] == self.n_way:
                feature_relu = nn.ReLU()
                init_weight = nn.functional.normalize(feature_relu(z_support),2,1) # L2 norm
                init_bias = torch.zeros(self.n_way).float()
                linear_clf.weight.data = init_weight
                linear_clf.bias.data = init_bias
            elif z_support.shape[0] == self.n_way * self.n_support:
                z_mean = torch.mean(torch.reshape(z_support, (self.n_way,self.n_support,-1)),dim=1)
                # print(z_mean.shape)
                feature_relu = nn.ReLU()
                init_weight = nn.functional.normalize(feature_relu(z_mean),2,1) # L2 norm
                init_bias = torch.zeros(self.n_way).float()
                linear_clf.weight.data = init_weight
                linear_clf.bias.data = init_bias
    
        linear_clf = linear_clf.cuda()

        
        set_optimizer = torch.optim.SGD(linear_clf.parameters(), lr = 1e-2, momentum=0.9, dampening=0.9, weight_decay=0.001)

        loss_function = nn.CrossEntropyLoss()
        loss_function = loss_function.cuda()

        batch_size = 4
        if temporal_aug:
            support_size = self.n_way* self.n_support* T
        else:
            support_size = self.n_way* self.n_support
        
        for epoch in range(100):
            rand_id = np.random.permutation(support_size)
            for i in range(0, support_size , batch_size):
                set_optimizer.zero_grad()
                selected_id = torch.from_numpy( rand_id[i: min(i+batch_size, support_size) ]).cuda()
                z_batch = z_support[selected_id]
                y_batch = y_support[selected_id]
                if self.loss_type == 'support':
                    z_relu = nn.ReLU().cuda()
                    z_batch = nn.functional.normalize(z_relu(z_batch),2,1) # L2 norm
                    
                scores = linear_clf(z_batch)
                loss = loss_function(scores,y_batch)
                loss.backward()
                set_optimizer.step()
            
        scores = linear_clf(z_query)
        pred = scores.data.cpu().numpy().argmax(axis = 1)
        y = np.repeat(range( self.n_way ), self.n_query )
        acc = np.mean(pred == y)*100
        return scores, acc


    def set_forward_loss(self,x):
        raise ValueError('Baseline predict on pretrained feature and do not support finetune backbone')

def feature_evaluation(cl_data_file, model, n_way = 5, n_support = 5, n_query = 1, adaptation = False, temporal_aug = False):
    class_list = cl_data_file.keys()

    select_class = random.sample(class_list,n_way)
    z_all  = []
    for cl in select_class:
        img_feat = cl_data_file[cl]
        perm_ids = np.random.permutation(len(img_feat)).tolist()
        z_all.append( [ np.squeeze( img_feat[perm_ids[i]]) for i in range(n_support+n_query) ] )     # stack each batch

    z_all = torch.from_numpy(np.array(z_all) )
   
    model.n_query = n_query
    if adaptation:
        scores,acc  = model.set_forward_adaptation(z_all, is_feature = True, temporal_aug = temporal_aug)
    else:
        scores  = model.set_forward(z_all, is_feature = True, temporal_aug = temporal_aug)
    # pred = scores.data.cpu().numpy().argmax(axis = 1)
    if temporal_aug:
        # y = np.repeat(range( n_way ), n_query* z_all.shape[-2] )
        soft_max = nn.Softmax(dim=1)
        pred = soft_max(scores).data.cpu().numpy()
        pred = np.average(pred.reshape(n_way, n_query, z_all.shape[-2], n_way),axis=2)
        pred = pred.reshape(n_way* n_query, n_way).argmax(axis = 1)
        y = np.repeat(range( n_way ), n_query )
        
    return acc


