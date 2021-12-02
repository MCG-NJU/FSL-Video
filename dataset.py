# import mmcv
import decord
from PIL import Image, ImageEnhance
import torch
import numpy as np
import torchvision.transforms as transforms
import json
import os

identity = lambda x:x
transformtypedict=dict(Brightness=ImageEnhance.Brightness, Contrast=ImageEnhance.Contrast, Sharpness=ImageEnhance.Sharpness, Color=ImageEnhance.Color)

class SubVdieoDataset:
    def __init__(self, sub_meta, cl, transform=transforms.ToTensor(), target_transform=identity, random_select=False, num_segments=None):
        self.sub_meta = sub_meta
        # self.video_list = [x.strip().split(' ') for x in open(sub_meta)]
        if True:
            self.image_tmpl = 'img_{:05d}.jpg'
        else:
            self.image_tmpl = 'img_{:05d}.png'
        self.cl = cl 
        self.transform = transform
        self.target_transform = target_transform
        self.random_select = random_select
        self.num_segments = num_segments
    
    def __getitem__(self,i):
        # image_path = os.path.join( self.sub_meta[i])
        assert len(self.sub_meta[i]) == 2
        full_path = self.sub_meta[i][0]
        num_frames = self.sub_meta[i][1]
        num_segments = self.num_segments
        if self.random_select and num_frames>8 : # random sample
            # frame_id = np.random.randint(num_frames)
            average_duration = num_frames // num_segments
            frame_id = np.multiply(list(range(num_segments)), average_duration)
            frame_id = frame_id + np.random.randint(average_duration, size=num_segments)
        else:
            # frame_id = num_frames//2
            tick = num_frames / float(num_segments)
            frame_id = np.array([int(tick / 2.0 + tick * x) for x in range(num_segments)])
        frame_id = frame_id + 1 # idx >= 1

        img_group = []
        for k in range(self.num_segments):
            img_path = os.path.join(full_path,self.image_tmpl.format(frame_id[k]))
            img = Image.open(img_path)
            img = self.transform(img)
            img_group.append(img)
        img_group = torch.stack(img_group,0)
        target = self.target_transform(self.cl)
        # print('ok',image_path)
        return img_group, target

    def __len__(self):
        return len(self.sub_meta)
        
class SetDataManager:
    def __init__(self, image_size, n_way, n_support, n_query, num_segments, n_eposide =100):        
        super(SetDataManager, self).__init__()
        self.image_size = image_size
        self.n_way = n_way
        self.batch_size = n_support + n_query
        self.n_eposide = n_eposide

        self.trans_loader = TransformLoader(image_size)
        self.num_segments = num_segments

    def get_data_loader(self, data_file, aug): #parameters that would change on train/val set
        transform = self.trans_loader.get_composed_transform(aug)
        dataset = SetDataset( data_file , self.batch_size, transform, random_select=aug, num_segments=self.num_segments) # video
        sampler = EpisodicBatchSampler(len(dataset), self.n_way, self.n_eposide )  
        data_loader_params = dict(batch_sampler = sampler,  num_workers = 8, pin_memory = True)       
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
        return data_loader

class TransformLoader:
    def __init__(self, image_size, 
                 # normalize_param    = dict(mean= [0.485, 0.456, 0.406] , std=[0.229, 0.224, 0.225]),
                 normalize_param    = dict(mean= [0.376, 0.401, 0.431] , std=[0.224, 0.229, 0.235]),
                 jitter_param       = dict(Brightness=0.4, Contrast=0.4, Color=0.4)):
        self.image_size = image_size
        self.normalize_param = normalize_param
        self.jitter_param = jitter_param
    
    def parse_transform(self, transform_type):
        if transform_type=='ImageJitter':
            method = ImageJitter( self.jitter_param )
            return method
        method = getattr(transforms, transform_type)
        if transform_type=='RandomResizedCrop':
            return method(self.image_size) 
        elif transform_type=='CenterCrop':
            return method(self.image_size) 
        elif transform_type=='Resize':
            return method([int(self.image_size*1.15), int(self.image_size*1.15)])
        elif transform_type=='Normalize':
            return method(**self.normalize_param )
        else:
            return method()

    def get_composed_transform(self, aug = False):
        if aug:
            transform_list = ['RandomResizedCrop', 'ImageJitter', 'RandomHorizontalFlip', 'ToTensor', 'Normalize']
        else:
            transform_list = ['Resize','CenterCrop', 'ToTensor', 'Normalize']

        transform_funcs = [ self.parse_transform(x) for x in transform_list]
        transform = transforms.Compose(transform_funcs)
        return transform

class SetDataset: # frames
    def __init__(self, data_file, batch_size, transform, random_select=False, num_segments=None):
        # with open(data_file, 'r') as f:
            # self.meta = json.load(f)
        self.video_list = [x.strip().split(' ') for x in open(data_file)]

        # self.cl_list = np.unique(self.meta['image_labels']).tolist()
        self.cl_list = np.zeros(len(self.video_list),dtype=int)
        for i in range(len(self.video_list)):
            self.cl_list[i] = self.video_list[i][2]
        self.cl_list = np.unique(self.cl_list).tolist()

        self.sub_meta = {}
        for cl in self.cl_list:
            self.sub_meta[cl] = []

        # for x,y in zip(self.meta['image_names'],self.meta['image_labels']):
            # self.sub_meta[y].append(x)
        for x in range(len(self.video_list)):
            root_path = self.video_list[x][0]
            num_frames = int(self.video_list[x][1])
            label = int(self.video_list[x][2])
            self.sub_meta[label].append([root_path,num_frames])

        self.sub_dataloader = [] 
        sub_data_loader_params = dict(batch_size = batch_size,
                                  shuffle = True,
                                  num_workers = 0, #use main thread only or may receive multiple batches
                                  pin_memory = False)        
        for cl in self.cl_list:
            sub_dataset = SubVdieoDataset(self.sub_meta[cl], cl, transform = transform ,random_select = random_select, num_segments=num_segments)
            self.sub_dataloader.append( torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params) )

    def __getitem__(self,i):
        return next(iter(self.sub_dataloader[i]))

    def __len__(self):
        return len(self.cl_list)

class EpisodicBatchSampler(object):
    def __init__(self, n_classes, n_way, n_episodes):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            yield torch.randperm(self.n_classes)[:self.n_way]

class ImageJitter(object):
    def __init__(self, transformdict):
        self.transforms = [(transformtypedict[k], transformdict[k]) for k in transformdict]


    def __call__(self, img):
        out = img
        randtensor = torch.rand(len(self.transforms))

        for i, (transformer, alpha) in enumerate(self.transforms):
            r = alpha*(randtensor[i]*2.0 -1.0) + 1
            out = transformer(out).enhance(r).convert('RGB')

        return out
