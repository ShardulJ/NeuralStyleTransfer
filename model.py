import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torchvision.models import vgg19
"""
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

Path_content = './chicago.jpg'
device = "cpu"

transform = transforms.Compose([
    transforms.Resize((256,256)),  # scale imported image
    transforms.ToTensor()])

to_pil = transforms.ToPILImage()

def load_image(image_name):
    image = Image.open(image_name).convert('RGB')
    image = transform(image).unsqueeze(0)
    return image.to(device)

c = load_image(Path_content)
"""

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

"""
if __name__ == "__main__":
	con = VGG19(content=True)
	out = con(c)
	print(out)
"""
