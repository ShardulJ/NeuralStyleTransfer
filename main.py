import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torchvision.models import vgg19

import os
import argparse
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from model import *

#transform = transforms.Compose([transforms.Resize((256,256)), transforms.ToTensor()])
device = "cpu"

def load_image(image_name,transform):
	#This function loads the images.
    image = Image.open(image_name)
    image = transform(image).unsqueeze(0)
    return image.to(device)
def convert(image):
    x = image.to("cpu").clone().detach().numpy().squeeze()
    x = x.transpose(1,2,0)
    x = x*np.array((0.485, 0.456, 0.406)) + np.array((0.229, 0.224, 0.225))
    return np.clip(x,0,1)

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

def gram_matrix(imgfeature):
    _,d,h,w = imgfeature.size()
    imgfeature = imgfeature.view(d,h*w)
    gram_mat = torch.mm(imgfeature,imgfeature.t())    
    return gram_mat

def get_args_parser():
    parser = argparse.ArgumentParser('Neural Style Transfer',add_help=False)
    parser.add_argument("--content_image", type=str, required=True, help="path to content image")
    parser.add_argument("--style_image", type=str, required=True, help="path to style image")
    parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs")
    parser.add_argument("--image_size", type=int, default=256, help="Size of images")
    parser.add_argument("--content_coeff", type=float, default=100, help="Weight for content loss")
    parser.add_argument("--style_coeff", type=float, default=1e8, help="Weight for style loss")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    return parser

def main(args):
    
    #Path to the images
    Path_style = args.content_image
    Path_content = args.style_image

    device = "cpu"

    transform = transforms.Compose([transforms.Resize((256,256)), transforms.ToTensor()])

    #it works like transpose(1,2,0)
    to_pil = transforms.ToPILImage()

    #load the images
    content_image = load_image(Path_content,transform=transform)
    style_image = load_image(Path_style,transform=transform)

    #load the pretrained model
    cnn = models.vgg19(pretrained=True).features
    model = VGG19(cnn).to(device)

    #Style-Features and gram matrix of style features
    style_features = model(style_image)
    style_grams = {y:gram_matrix(style_features[y]) for y in style_features}
    
    #Content-Features
    content_features = model(content_image)
    
    #clone the traget image
    target_image = content_image.clone().requires_grad_(True).to(device)

    #parameters
    print_after = 100
    epochs = args.epochs
    style_coeff = args.style_coeff
    content_coeff = args.content_coeff

    #styleLayers
    style_layers = ["conv_1_1", "conv_2_1", "conv_3_1", "conv_4_1", "conv_5_1"]
    
    #optimizer
    optimizer = torch.optim.Adam([target_image],lr=args.lr)

    #defining the MSE Loss
    l2_loss = torch.nn.MSELoss().to(device)

    for i in range(epochs):
    
        target_features = model(target_image)

        # Compute content loss as MSE between features
        content_loss = l2_loss(content_features['conv_4_2'],target_features['conv_4_2'])

        # Compute style loss as MSE between gram matrices
        style_loss = 0
        for layer in style_layers:
            style_gram = style_grams[layer]
            target_gram = target_features[layer]
            target_gram = gram_matrix(target_gram)
            style_loss += l2_loss(target_gram,style_gram)
        style_loss *= style_coeff
        content_loss *= content_coeff

        total_loss = style_loss + content_loss

        if i%10==0:       
            print("epoch ",i," ", total_loss)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        if i%print_after == 0:
        plt.imshow(convert(target_image),label="Epoch "+str(i))
        plt.show()
        plt.imsave('./output/'+str(i)+'.png',convert(target_image),format='png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Parser for neural style transfer', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)

