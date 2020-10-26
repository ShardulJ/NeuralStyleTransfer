import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class VGG19(nn.Module):
    
    #device = "cpu"

    def __init__(self,cnn):
        super(VGG19, self).__init__()
        self.cnn = copy.deepcopy(cnn)
        
        #Hack to set relu inplace=False
        self.model = nn.Sequential()
        i = 0
        for layer in self.cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                # The in-place version doesn't play very nicely with the ContentLoss
                # and StyleLoss we insert below. So we replace with out-of-place
                # ones here.
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            self.model.add_module(name, layer)
        
        for p in self.model.parameters():
            p.requires_grad = False
        
        self.model.to("cpu")
        self.conv_1_1 = nn.Sequential(*list(self.model.children())[0:1])
        self.conv_2_1 = nn.Sequential(*list(self.model.children())[1:6])
        self.conv_3_1 = nn.Sequential(*list(self.model.children())[6:11])   
        self.conv_4_1 = nn.Sequential(*list(self.model.children())[11:20])
        self.conv_4_2 = nn.Sequential(*list(self.model.children())[11:22])
        self.conv_5_1 = nn.Sequential(*list(self.model.children())[20:29])

    
    def forward(self, x):
        out1 = self.conv_1_1(x)
        out_1_1 = out1.clone()
        out2 = self.conv_2_1(out1)
        out_2_1 = out2.clone()
        out3 = self.conv_3_1(out2)
        out_3_1 = out3.clone()
        out4 = self.conv_4_1(out3)
        out_4_1 = out4.clone()
        out4x = self.conv_4_2(out3)
        out_4_2 = out4x.clone()
        out5 = self.conv_5_1(out4)
        out_5_1 = out5.clone()
        features = {"conv_1_1": out_1_1, "conv_2_1" : out_2_1, "conv_3_1" : out_3_1, "conv_4_1" : out_4_1, "conv_5_1" : out_5_1, "conv_4_2" : out_4_2}
        return features
