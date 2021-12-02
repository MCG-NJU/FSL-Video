import torch
import torch.nn as nn
import sys

def ResNet50_pretrained():
    import torchvision.models as models
    new_model = models.resnet50(pretrained=True)
    new_model.fc = nn.Identity()
    new_model.final_feat_dim = 2048
    return new_model

def ResNet50_model():
    import torchvision.models as models
    new_model = models.resnet50(pretrained=False)
    new_model.fc = nn.Identity()
    new_model.final_feat_dim = 2048
    return new_model

class Identity(torch.nn.Module):
    def forward(self, input):
        return input

def ResNet50_tam_pretrained(num_class, num_segments, tam):
    sys.path.append('/home/zzx/workspace/code/temporal-adaptive-module/')
    from ops.models import TSN
    new_model = TSN(num_class, num_segments, 'RGB', 'resnet50', dropout=0.5, tam = tam, partial_bn=False)
    new_model.new_fc = Identity()
    new_model.softmax = Identity()
    new_model.final_feat_dim = 2048
    return new_model

def ResNet50_moco(checkpoint_path = '/home/zzx/workspace/data/pretrained_models/moco_v1_200ep_pretrain.pth.tar'):
    import torchvision.models as models
    new_model = models.resnet50(pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['state_dict']
    for k in list(state_dict.keys()):
        # retain only encoder_q up to before the embedding layer
        if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
            # remove prefix
            state_dict[k[len("module.encoder_q."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]

    msg = new_model.load_state_dict(state_dict, strict=False)
    assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
    new_model.fc = nn.Identity()
    new_model.final_feat_dim = 2048
    return new_model



