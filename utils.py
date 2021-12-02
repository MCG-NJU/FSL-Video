import argparse
import backbone

model_dict = dict(
            ResNet50_pretrained = backbone.ResNet50_pretrained,
            ResNet50_model = backbone.ResNet50_model,
            ResNet50_tam_pretrained = backbone.ResNet50_tam_pretrained,
            ResNet50_moco = backbone.ResNet50_moco)

def parse_args(script):
    parser = argparse.ArgumentParser(description= 'few-shot script %s' %(script))
    parser.add_argument('--dataset'     , default='somethingotam')
    parser.add_argument('--model'       , default='ResNet50_pretrained') 
    parser.add_argument('--train_n_way' , default=5, type=int)
    parser.add_argument('--test_n_way'  , default=5, type=int) 
    parser.add_argument('--n_shot'      , default=1, type=int) 
    parser.add_argument('--train_aug'   , default=True, type=bool)
    parser.add_argument('--work_dir'     , default='/home/zzx/workspace2/FSL/proto')
    parser.add_argument('--num_segments', default=8, type=int)
    parser.add_argument('--n_query', default=1, type=int)
    parser.add_argument('--eval_freq', default=10, type=int)
    parser.add_argument('--eval_episode', default=1000, type=int)
    parser.add_argument('--test_episode', default=10000, type=int)
    parser.add_argument('--test_model', default=False, type=bool)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--save_freq'   , default=100, type=int)
    parser.add_argument('--start_epoch' , default=0, type=int)
    parser.add_argument('--stop_epoch'  , default=-1, type=int)
    parser.add_argument('--lr', default = 1e-5, type=float)
    
    return parser.parse_args()