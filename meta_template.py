import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from abc import abstractmethod
import math

class MetaTemplate(nn.Module):
    def __init__(self, model_func, n_way, n_support):
        super(MetaTemplate, self).__init__()
        self.n_way      = n_way
        self.n_support  = n_support
        self.n_query    = 1 #(change depends on input) 
        self.feature    = model_func()
        self.feat_dim   = self.feature.final_feat_dim
    
    @abstractmethod
    def forward(self,x):
        pass

    def parse_feature(self,x,is_feature):
        x    = Variable(x.cuda())
        if is_feature:
            z_all = x
        else:
            x = x.contiguous().view( self.n_way * (self.n_support + self.n_query), *x.size()[2:]) 
            assert len(x.shape) == 5 
            z_all = 0
            for i in range(x.shape[1]):
                z_all += self.feature.forward(x[:,i,:,:,:]) 
            z_all = z_all / x.shape[1]
            
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

class BaseNet(MetaTemplate):
    def __init__(self, model_func,  n_way, n_support):
        super(BaseNet, self).__init__( model_func,  n_way, n_support)

    def forward(self,x):
        return self.feature.forward(x)

class CMN(MetaTemplate):
    def __init__(self, model_func,  n_way, n_support):
        super(CMN, self).__init__( model_func,  n_way, n_support)
        self.loss_fn = nn.CrossEntropyLoss()
        self.num_frames = 8
        self.sqrtd = math.sqrt(self.feat_dim)
        # self.h = torch.rand((self.num_frames,self.feat_dim),requires_grad=True).cuda() # 8 x 2048
        self.h = nn.Parameter(torch.full((self.num_frames,self.feat_dim), 1e-8, requires_grad=True).cuda()) # average init
        self.fc = nn.Linear(self.num_frames*self.feat_dim,self.feat_dim)
        fc_weight = np.diag([1]*self.feat_dim)
        fc_weight = np.tile(fc_weight, self.num_frames)
        self.fc.weight.data = torch.from_numpy(fc_weight).float() # average init
        torch.nn.init.constant_(self.fc.bias, 0.0)

    def forward(self,x):
        return self.feature.forward(x) 

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
