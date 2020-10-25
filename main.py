import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, models
from torchvision.models import vgg19

import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

#Path to the images
Path_style = './artistic_style.jpeg'
Path_content = './chicago.jpg'

device = "cpu"

transform = transforms.Compose([transforms.Resize((256,256)), transforms.ToTensor()])

#it works like transpose(1,2,0)
to_pil = transforms.ToPILImage()

def load_image(image_name):
	#This function loads the images.
    image = Image.open(image_name)
    image = transform(image)
    return image.to(device)

#load the images
c = load_image(Path_content)
s = load_image(Path_style)

def plot_img(content_img, style_img, title=None):
	#Function plots both the images.
	
    content_image = to_pil(content_img.cpu().clone().squeeze())
    
    style_image = to_pil(style_img.cpu().clone().squeeze())
    
    fig = plt.figure(figsize = (20, 10))
    
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(content_image, aspect="auto")
    if title is not None:
        plt.title(title[0], fontsize=30)
    ax.axis('off')
    
    ax = fig.add_subplot(1, 2, 2)
    ax.imshow(style_image, aspect="auto")
    if title is not None:
        ax.set_title(title[1], fontsize=30)
    ax.axis('off')

#load the pretrained model
#model = models.vgg19(pretrained=True).features

if __name__ == "__main__":
	plot_img(c, s, title=['Content-Image', 'Style-Image'])
	plt.show()




















