import torch
import torch.nn as nn
from torchvision import models
from collections import OrderedDict

# model_path = './tracking/weights/resnet50_berry_add_2.pth'
model_path = 'resnet50_berry_add_1.pt'
# model_path = 'net_last.pth'

state_dict = torch.load(model_path, map_location=torch.device('cuda'))
print(state_dict)
# 读取模型

new_state_dict = OrderedDict()
for k, v in state_dict.items():
    if k.startswith("_orig_mod."):
        new_key = k.replace('_orig_mod.', '')
    elif k.startswith("_orig_mod.model."):
        new_key = k.replace('_orig_mod.model.', '')
    elif k.startswith("model."):
        new_key = k.replace('model.', '')
    # elif k.startswith("fc."):
    #     continue
    elif k.startswith("classifier.add_block"):
        new_key = k.replace('classifier.add_block', 'fc')
    elif k.startswith("classifier.classifier"):
        new_key = k.replace('classifier.classifier.0.', 'classifier.')
    else:
        new_key = k
    new_state_dict[new_key] = v
print(new_state_dict)
torch.save(new_state_dict, "resnet50_berry_add_1.pt")




