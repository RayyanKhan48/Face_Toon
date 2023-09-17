'''
Testing final model
Team 29
Nov30 2021

'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, utils
import numpy as np
import os
import matplotlib.pyplot as plt
import torchvision.models

'''
Parameters
'''
torch.backends.cudnn.benchmark = True
dtype = torch.cuda.FloatTensor
torch.manual_seed(24)
'''
Data processing
'''
image_transform = transforms.Compose([
                                transforms.Resize(64),
                                transforms.CenterCrop(64),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                
                              
                              ])
'''
Setting up the custom dataset class
'''
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, folder_path, transform):
        self.path = folder_path
        self.file_path = os.listdir(self.path)
        self.data = []
        self.transform = transform
        for file_paths in self.file_path:
            image = plt.imread(os.path.join(self.path, file_paths))
            self.data.append(np.array(image))
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        self.intToFloat = torch.nn.Sequential(transforms.ConvertImageDtype(torch.float32))
        image = self.intToFloat(torch.from_numpy(self.data[idx]).transpose(0,2))
        return self.transform(image)

def showImg(image):
    figure = plt.figure(figsize=(3,3))
    figure.add_subplot(1,1,1)
    normalized_img = image * 0.5 + 0.5
    plt.imshow(normalized_img.transpose(0,2))


'''
Generator

Note that not all layer are used in forward, some are kept to be compatible with older saved state.
'''
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, 1, 1, bias = False)
        self.conv2 = nn.Conv2d(8, 16, 5, 2, 2, bias = False)
        self.conv3 = nn.Conv2d(16, 32, 6, 2, 2, bias = False)

        self.convt1 = nn.ConvTranspose2d(32, 16, 6, 2, 2, bias = False)
        self.convt2 = nn.ConvTranspose2d(16, 8, 4, 2, 2, bias = False)
        self.convt3 = nn.ConvTranspose2d(8, 3, 3, 1, 0, bias = False)
        
        self.c1 = nn.Conv2d(3, 12, 3, 1, 1, bias = False)
        self.c2 = nn.Conv2d(12, 24, 4, 2, 1, bias = False)
        self.c3 = nn.ConvTranspose2d(24, 12, 4, 2, 1, bias = False)
        self.c4 = nn.ConvTranspose2d(12, 3, 3, 1, 1, bias = False)

        self.c5 = nn.Conv2d(3, 3, 3, 1, 1, bias = False)
        self.cnorm1 = nn.BatchNorm2d(12)
        self.cnorm2 = nn.BatchNorm2d(24)

        self.norm1 = nn.BatchNorm2d(8)
        self.norm2 = nn.BatchNorm2d(16)
        self.norm3 = nn.BatchNorm2d(32)
        self.norm4 = nn.BatchNorm2d(16)
        self.norm5 = nn.BatchNorm2d(8)
        self.norm6 = nn.BatchNorm2d(3)
        
        self.drop = nn.Dropout(p=0.12)
    def forward(self, x):

        x = F.relu(self.norm1(self.conv1(x)))
        x = self.drop(x)
        
        x = F.relu(self.norm2(self.conv2(x)))
        x = self.drop(x)

        x = F.relu(self.norm3(self.conv3(x)))
        x = self.drop(x)
        
        x = F.relu(self.norm4(self.convt1(x)))
        
        x = F.relu(self.norm5(self.convt2(x)))
        
        x = F.tanh(self.convt3(x))
        
        return x

def showTensorImg(net, tensor):
    showImg(net(tensor.cuda().unsqueeze(0)).detach().cpu()[0])

'''
Model
'''   
GEN_A = Generator().cuda()
GEN_B = Generator().cuda()

print("DONE")

'''
Loading in the model state and putting them into evaluation mode.
'''
GEN_A.load_state_dict(torch.load("/content/GA_save_4152"))
GEN_A.eval()
GEN_B.load_state_dict(torch.load("/content/GB_save_4152"))
GEN_B.eval()

"""Loading in data for testing"""

#Data path should be a folder containing all the images you want for testing
#You do not have to load anime faces if you only want to convert human to anime
#Run this cell if you want to setup anime to human faces conversion
anime_data = ImageDataset("/content/anime", image_transform)
print("anime data loaded...")

#Run this cell if you want to setup human to anime faces conversion
human_data = ImageDataset("/content/human", image_transform)
print("human data loaded...")

"""Showing Human faces to anime faces"""

'''
Generating output using our Generator B with testing data
'''

#Showing unselected output of our model using testing data from flicker HQ.
#These images are diverse in facial expression, races, genders, background and ages. Thus hard to convert.
for j in range(24): #range determines amount of faces shown.
    i = j+24 #offset for data number, For example, +40 and range(5) will give data number 40 to 45
    showImg(human_data[i])
    showTensorImg(GEN_B, human_data[i])

"""Showing anime faces to human faces"""

'''
Generating output using our Generator A with testing data
Note: This was not the purpose of our model so Generator A has worse performance than generator B since our hyperparamter is not tuned
for generator A.
'''

#We can invetigate the latent space of our generator with this code.
for j in range(10):
    i = j + 15
    showImg(anime_data[i])
    showTensorImg(GEN_A, anime_data[i])