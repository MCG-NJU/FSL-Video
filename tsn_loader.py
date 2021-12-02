# import mmcv
import numpy as np
import os.path as osp
import os
import json
import torch
# from mmcv.parallel import DataContainer as DC
from torch.utils.data import Dataset
from collections import Sequence
from PIL import Image, ImageEnhance

# from transforms import (GroupImageTransform)
# from .utils import to_tensor
identity = lambda x:x

try:
    import decord
except ImportError:
    pass


def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.
    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.
    """
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not mmcv.is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError('type {} cannot be converted to tensor.'.format(
            type(data)))


class RawFramesRecord(object):

    def __init__(self, path, label):
        self.path = path
        self.label = int(label)
        self.num_frames = -1

class SimpleVideoDataset:
    def __init__(self, data_file, transform, target_transform=identity, random_select=False, num_segments=None):
        '''
        with open(data_file, 'r') as f:
            self.meta = json.load(f)
        '''
        self.video_list = [x.strip().split(' ') for x in open(data_file)]
        self.image_tmpl = 'img_{:05d}.jpg'
         
        self.transform = transform
        self.target_transform = target_transform
        self.random_select = random_select
        self.num_segments = num_segments

    def __getitem__(self,i):
        full_path = self.video_list[i][0]
        num_frames = int(self.video_list[i][1])
        num_segments = self.num_segments
        if self.random_select and num_frames>8 : # random sample
            average_duration = num_frames // num_segments
            frame_id = np.multiply(list(range(num_segments)), average_duration)
            frame_id = frame_id + torch.randint(average_duration, size=(num_segments,)).numpy()
        else:
            tick = num_frames / float(num_segments)
            frame_id = np.array([int(tick / 2.0 + tick * x) for x in range(num_segments)])
        frame_id = frame_id + 1 # idx >= 1
        img_group = []
        for k in range(self.num_segments):
            # img = video_reader[frame_id[k]].asnumpy() # 256 340 3
            # convert to PIL
            img_path = os.path.join(full_path,self.image_tmpl.format(frame_id[k]))
            img = Image.open(img_path)
            img = self.transform(img)
            img_group.append(img)
        img_group = torch.stack(img_group,0)
        target = self.target_transform(int(self.video_list[i][2]))
        # print('ok',image_path)
        return img_group, target
    ''' # from video
    def __getitem__(self,i):
        #print( '%d -%d' %(self.cl,i))
        image_path = os.path.join(self.meta['image_names'][i])
        # video_reader = mmcv.VideoReader(image_path)
        video_reader = decord.VideoReader(image_path)
        num_frames = len(video_reader)
        num_segments = self.num_segments
        if self.random_select:
            # frame_id = np.random.randint(num_frames)
            average_duration = num_frames // num_segments
            frame_id = np.multiply(list(range(num_segments)), average_duration)
            # frame_id = frame_id + np.random.randint(average_duration, size=num_segments)
            frame_id = frame_id + torch.randint(average_duration, size=(num_segments,)).numpy()
        else:
            # frame_id = num_frames//2
            tick = num_frames / float(num_segments)
            frame_id = np.array([int(tick / 2.0 + tick * x) for x in range(num_segments)])
        
        img_group = []
        for k in range(self.num_segments):
            img = video_reader[frame_id[k]].asnumpy() # 256 340 3
            # convert to PIL
            img = Image.fromarray(img)
            img = self.transform(img)
            img_group.append(img)
        img_group = torch.stack(img_group,0)
        target = self.target_transform(self.meta['image_labels'][i])
        # print('ok',image_path)
        return img_group, target
    '''
    def __len__(self):
        # return len(self.meta['image_names'])
        return len(self.video_list)

class SimpleDataManager:
    def __init__(self, image_size, batch_size, num_segments=None):        
        super(SimpleDataManager, self).__init__()
        self.batch_size = batch_size
        self.num_segments = num_segments
        self.trans_loader = TransformLoader(image_size)

    def get_data_loader(self, data_file, aug): #parameters that would change on train/val set
        transform = self.trans_loader.get_composed_transform(aug)
        # dataset = SimpleDataset(data_file, transform)
        dataset = SimpleVideoDataset(data_file, transform, random_select=aug, num_segments=self.num_segments)
        data_loader_params = dict(batch_size = self.batch_size, shuffle = True, num_workers = 8, pin_memory = True)       
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)

        return data_loader

import torchvision.transforms as transforms
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

transformtypedict=dict(Brightness=ImageEnhance.Brightness, Contrast=ImageEnhance.Contrast, Sharpness=ImageEnhance.Sharpness, Color=ImageEnhance.Color)
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

class VideoDataset(Dataset):

    def __init__(self,
                 data_file,
                 img_prefix,
                 img_norm_cfg,
                 num_segments=3,
                 new_length=1,
                 new_step=1,
                 random_shift=True,
                 temporal_jitter=False,
                 modality='RGB',
                 image_tmpl='img_{}.jpg',
                 img_scale=256,
                 img_scale_file=None,
                 input_size=224,
                 div_255=False,
                 size_divisor=None,
                 proposal_file=None,
                 num_max_proposals=1000,
                 flip_ratio=0.5,
                 resize_keep_ratio=True,
                 resize_ratio=[1, 0.875, 0.75, 0.66],
                 test_mode=False,
                 oversample=None,
                 random_crop=False,
                 more_fix_crop=False,
                 multiscale_crop=False,
                 resize_crop=False,
                 rescale_crop=False,
                 scales=None,
                 max_distort=1,
                 input_format='NCHW',
                 use_decord=False):
        # prefix of images path
        self.img_prefix = img_prefix

        # load annotations
        # self.video_infos = self.load_annotations(ann_file)
        with open(data_file, 'r') as f:
            self.meta = json.load(f)

        # normalization config
        self.img_norm_cfg = img_norm_cfg

        # parameters for frame fetching
        # number of segments
        self.num_segments = num_segments
        # number of consecutive frames
        self.old_length = new_length * new_step
        self.new_length = new_length
        # number of steps (sparse sampling for efficiency of io)
        self.new_step = new_step
        # whether to temporally random shift when training
        self.random_shift = random_shift
        # whether to temporally jitter if new_step > 1
        self.temporal_jitter = temporal_jitter

        # parameters for modalities
        if isinstance(modality, (list, tuple)):
            self.modalities = modality
            num_modality = len(modality)
        else:
            self.modalities = [modality]
            num_modality = 1
        if isinstance(image_tmpl, (list, tuple)):
            self.image_tmpls = image_tmpl
        else:
            self.image_tmpls = [image_tmpl]
        assert len(self.image_tmpls) == num_modality

        # parameters for image preprocessing
        # img_scale
        if isinstance(img_scale, int):
            img_scale = (np.Inf, img_scale)
        self.img_scale = img_scale
        if img_scale_file is not None:
            self.img_scale_dict = {
                line.split(' ')[0]:
                (int(line.split(' ')[1]), int(line.split(' ')[2]))
                for line in open(img_scale_file)
            }
        else:
            self.img_scale_dict = None
        # network input size
        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        self.input_size = input_size

        # parameters for specification from pre-trained networks (lecacy issue)
        self.div_255 = div_255

        # parameters for data augmentation
        # flip ratio
        self.flip_ratio = flip_ratio
        self.resize_keep_ratio = resize_keep_ratio

        # test mode or not
        self.test_mode = test_mode

        # set group flag for the sampler
        # if not self.test_mode:
        self._set_group_flag()

        # transforms
        assert oversample in [None, 'three_crop', 'ten_crop']
        self.img_group_transform = GroupImageTransform(
            size_divisor=None,
            crop_size=self.input_size,
            oversample=oversample,
            random_crop=random_crop,
            more_fix_crop=more_fix_crop,
            multiscale_crop=multiscale_crop,
            scales=scales,
            max_distort=max_distort,
            resize_crop=resize_crop,
            rescale_crop=rescale_crop,
            **self.img_norm_cfg)

        # input format
        assert input_format in ['NCHW', 'NCTHW']
        self.input_format = input_format
        '''
        self.bbox_transform = Bbox_transform()
        '''

        self.use_decord = use_decord
        # self.video_ext = video_ext

    def __len__(self):
        return len(self.meta['image_names'])

    def load_annotations(self, ann_file):
        return [RawFramesRecord(x.strip().split(' ')) for x in open(ann_file)]
        # return mmcv.load(ann_file)

    def load_proposals(self, proposal_file):
        return mmcv.load(proposal_file)

    def get_ann_info(self, idx):
        return {
            'path': self.video_infos[idx].path,
            'label': self.video_infos[idx].label
        }
        # return self.video_infos[idx]['ann']

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.
        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            # img_info = self.img_infos[i]
            # if img_info['width'] / img_info['height'] > 1:
            self.flag[i] = 1

    def _load_image(self, video_reader, directory, modality, idx):
        if modality in ['RGB', 'RGBDiff']:
            return [video_reader[idx - 1]]
        elif modality == 'Flow':
            raise NotImplementedError
        else:
            raise ValueError('Not implemented yet; modality should be '
                             '["RGB", "RGBDiff", "Flow"]')

    def _sample_indices(self, record):
        '''
        :param record: VideoRawFramesRecord
        :return: list, list
        '''
        average_duration = (record.num_frames - self.old_length +
                            1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(
                list(range(self.num_segments)), average_duration)
            offsets = offsets + np.random.randint(
                average_duration, size=self.num_segments)
        elif record.num_frames > max(self.num_segments, self.old_length):
            offsets = np.sort(
                np.random.randint(
                    record.num_frames - self.old_length + 1,
                    size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments, ))
        if self.temporal_jitter:
            skip_offsets = np.random.randint(
                self.new_step, size=self.old_length // self.new_step)
        else:
            skip_offsets = np.zeros(
                self.old_length // self.new_step, dtype=int)
        return offsets + 1, skip_offsets  # frame index starts from 1

    def _get_val_indices(self, record):
        if record.num_frames > self.num_segments + self.old_length - 1:
            tick = (record.num_frames - self.old_length + 1) / \
                float(self.num_segments)
            offsets = np.array(
                [int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments, ))
        if self.temporal_jitter:
            skip_offsets = np.random.randint(
                self.new_step, size=self.old_length // self.new_step)
        else:
            skip_offsets = np.zeros(
                self.old_length // self.new_step, dtype=int)
        return offsets + 1, skip_offsets

    def _get_test_indices(self, record):
        if record.num_frames > self.old_length - 1:
            tick = (record.num_frames - self.old_length + 1) / \
                float(self.num_segments)
            offsets = np.array(
                [int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments, ))
        if self.temporal_jitter:
            skip_offsets = np.random.randint(
                self.new_step, size=self.old_length // self.new_step)
        else:
            skip_offsets = np.zeros(
                self.old_length // self.new_step, dtype=int)
        return offsets + 1, skip_offsets

    def _get_frames(self, record, video_reader, image_tmpl, modality, indices,
                    skip_offsets):
        if self.use_decord:
            if modality not in ['RGB', 'RGBDiff']:
                raise NotImplementedError
            images = list()
            for seg_ind in indices:
                p = int(seg_ind)
                if p > 1:
                    video_reader.seek(p - 1)
                cur_content = video_reader.next().asnumpy()
                # Cache the (p-1)-th frame first. This is to avoid decord's
                # StopIteration, which may consequently affect the mmcv.runner
                for i, ind in enumerate(
                        range(0, self.old_length, self.new_step)):
                    if (skip_offsets[i] > 0
                            and p + skip_offsets[i] <= record.num_frames):
                        if skip_offsets[i] > 1:
                            video_reader.skip_frames(skip_offsets[i] - 1)
                        cur_content = video_reader.next().asnumpy()
                    seg_imgs = [cur_content]
                    images.extend(seg_imgs)
                    if (self.new_step > 1
                            and p + self.new_step <= record.num_frames):
                        video_reader.skip_frames(self.new_step - 1)
                    p += self.new_step
            return images
        else:
            images = list()
            for seg_ind in indices:
                p = int(seg_ind)
                for i, ind in enumerate(
                        range(0, self.old_length, self.new_step)):
                    if p + skip_offsets[i] <= record.num_frames:
                        seg_imgs = self._load_image(
                            video_reader, osp.join(self.img_prefix,
                                                   record.path), modality,
                            p + skip_offsets[i])
                    else:
                        seg_imgs = self._load_image(
                            video_reader, osp.join(self.img_prefix,
                                                   record.path), modality, p)
                    images.extend(seg_imgs)
                    if p + self.new_step < record.num_frames:
                        p += self.new_step
            return images

    def __getitem__(self, idx):
        # record = self.video_infos[idx]
        path = osp.join(self.meta['image_names'][idx]) # video path
        target = self.meta['image_labels'][idx] # label
        record = RawFramesRecord(path, target)
        if self.use_decord:
            video_reader = decord.VideoReader(record.path)
            record.num_frames = len(video_reader)
        else:
            video_reader = mmcv.VideoReader(record.path)
            record.num_frames = len(video_reader)

        if self.test_mode:
            segment_indices, skip_offsets = self._get_test_indices(record)
        else:
            segment_indices, skip_offsets = self._sample_indices(
                record) if self.random_shift else self._get_val_indices(record)

        data = dict(
            num_modalities=DC(to_tensor(len(self.modalities))),
            gt_label=DC(to_tensor(record.label), stack=True, pad_dims=None))

        # handle the first modality
        modality = self.modalities[0]
        image_tmpl = self.image_tmpls[0]
        try:
            img_group = self._get_frames(record, video_reader, image_tmpl, modality, segment_indices, skip_offsets)
        except: # use mmcv instead
            self.use_decord == False
            # print(record.path)
            return self.__getitem__(idx)

        flip = True if np.random.rand() < self.flip_ratio else False
        if (self.img_scale_dict is not None
                and record.path in self.img_scale_dict):
            img_scale = self.img_scale_dict[record.path]
        else:
            img_scale = self.img_scale
        (img_group, img_shape, pad_shape, scale_factor,
         crop_quadruple) = self.img_group_transform(
             img_group,
             img_scale,
             crop_history=None,
             flip=flip,
             keep_ratio=self.resize_keep_ratio,
             div_255=self.div_255,
             is_flow=True if modality == 'Flow' else False)
        ori_shape = (256, 340, 3)
        img_meta = dict(
            ori_shape=ori_shape,
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            crop_quadruple=crop_quadruple,
            flip=flip)
        # [M x C x H x W]
        # M = 1 * N_oversample * N_seg * L
        if self.input_format == "NCTHW":
            img_group = img_group.reshape((-1, self.num_segments,
                                           self.new_length) +
                                          img_group.shape[1:])
            # N_over x N_seg x L x C x H x W
            img_group = np.transpose(img_group, (0, 1, 3, 2, 4, 5))
            # N_over x N_seg x C x L x H x W
            img_group = img_group.reshape((-1, ) + img_group.shape[2:])
            # M' x C x L x H x W

        data.update(
            dict(
                img_group_0=DC(to_tensor(img_group), stack=True, pad_dims=2),
                img_meta=DC(img_meta, cpu_only=True)))

        # handle the rest modalities using the same
        for i, (modality, image_tmpl) in enumerate(
                zip(self.modalities[1:], self.image_tmpls[1:])):
            img_group = self._get_frames(record, video_reader, image_tmpl, modality,
                                         segment_indices, skip_offsets)

            # apply transforms
            flip = True if np.random.rand() < self.flip_ratio else False
            (img_group, img_shape, pad_shape, scale_factor,
             crop_quadruple) = self.img_group_transform(
                 img_group,
                 img_scale,
                 crop_history=data['img_meta']['crop_quadruple'],
                 flip=data['img_meta']['flip'],
                 keep_ratio=self.resize_keep_ratio,
                 div_255=self.div_255,
                 is_flow=True if modality == 'Flow' else False)
            if self.input_format == "NCTHW":
                # Convert [M x C x H x W] to [M' x C x T x H x W]
                # M = 1 * N_oversample * N_seg * L
                # M' = 1 * N_oversample * N_seg, T = L
                img_group = img_group.reshape((-1, self.num_segments,
                                               self.new_length) +
                                              img_group.shape[1:])
                img_group = np.transpose(img_group, (0, 1, 3, 2, 4, 5))
                img_group = img_group.reshape((-1, ) + img_group.shape[2:])

            data.update({
                'img_group_{}'.format(i + 1):
                DC(to_tensor(img_group), stack=True, pad_dims=2),
            })
        # print(data)
        
        return data['img_group_0'].data, torch.squeeze(data['gt_label'].data)

if __name__ == "__main__":
    base_file = '/home/zzx/workspace/code/video_FSL/baseline/base.json'
    vid_path = '/home/zzx/workspace/data/kinetics_100/001.air_drumming/-fTmHyOG-CY.mp4'
    img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    err_vid_path = '/home/zzx/workspace/data/kinetics_100/064.weaving_basket/vcxx4Vy8jCM.mp4'
    video_reader = decord.VideoReader(err_vid_path)
    num_frames = len(video_reader)
    print(num_frames)
    for seg_ind in range(10000):
        p = int(seg_ind)
        if p > 1:
            video_reader.seek(p - 1)
        cur_content = video_reader.next().asnumpy()


    train_dataset = VideoDataset(data_file=base_file,img_prefix='/home/zzx/workspace/data/kinetics_100',img_norm_cfg=img_norm_cfg,num_segments=8)
    print(train_dataset.__len__())
    data_loader_params = dict(batch_size = 2, shuffle = True, num_workers = 1, pin_memory = True)       
    train_loader = torch.utils.data.DataLoader(train_dataset, **data_loader_params)

    # for i, (x,y) in enumerate(train_loader):
        # print(x.shape,y)
    for i, (x,y) in enumerate(train_loader):
        print(x.shape,y)