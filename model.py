import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torchvision.models import vgg19


class VGG19(nn.Module):
    def __init__(self, content=False):
        super(VGG19, self).__init__()
        self.model = vgg19(pretrained=True).features
        self.content = content
        self.conv1_1 = nn.Sequential(*list(self.model.children())[0:1])
        self.conv_2_1 = nn.Sequential(*list(self.model.children())[1:6])
        self.conv_3_1 = nn.Sequential(*list(self.model.children())[6:11])
        if self.content:
            self.conv_4_2 = nn.Sequential(*list(self.model.children())[11:22])
        self.conv_4_1 = nn.Sequential(*list(self.model.children())[11:20])
        self.conv_5_1 = nn.Sequential(*list(self.model.children())[20:29])


    def forward(self, x):
        if not self.content:
            out_1_1 = self.conv1_1(x)
            out_2_1 = self.conv_2_1(out_1_1)
            out_3_1 = self.conv_3_1(out_2_1)
            out_4_1 = self.conv_4_1(out_3_1)
            out_5_1 = self.conv_5_1(out_4_1)
            return [out_1_1, out_2_1, out_3_1, out_4_1, out_5_1]
        else:
            out_1_1 = self.conv1_1(x)
            out_2_1 = self.conv_2_1(out_1_1)
            out_3_1 = self.conv_3_1(out_2_1)
            out_4_2 = self.conv_4_2(out_3_1)
            return out_4_2

