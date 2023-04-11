#!/usr/bin/env python3

'''
 Abinav Anantharaman and Satwik Bhandiwad
   CS 5330 Spring 2023
   Recognition using Deep Networks 
'''

import cv2
import torch
import torchvision
from torchsummary import summary
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from digit_dataset_train import Net


torch.backends.cudnn.enabled = False
torch.manual_seed(42)

# Load the saved model
network = Net()
network.load_state_dict(torch.load('model.pth'))

# Show Network summary
summary(network, (1, 28, 28))
print(network)


# Analyze the first layer
weights = network.conv1.weight.detach()
print("Filter weights shape:", weights.shape)


testloader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('files/', train=False, download=True,
                                    transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                        (0.1307,), (0.3081,))
                                    ])),
        batch_size=10, shuffle=True)

images, labels = next(iter(testloader))

# Access the first image in the batch
first_image = images[0]
# Convert the tensor to a NumPy array
img = first_image.permute(1, 2, 0).numpy()

fig = plt.figure()
with torch.no_grad():
    for i in range(weights.shape[0]):
        plt.subplot(5,4,(i*2)+1)
        plt.tight_layout()
        plt.imshow(weights[i, 0], cmap='gray', interpolation='none')
        plt.title("Filter {}".format(i+1))
        plt.xticks([])
        plt.yticks([])

        plt.subplot(5,4,(i*2)+2)
        plt.tight_layout()
        filtered_img = cv2.filter2D(img, -1, weights[i, 0].numpy())
        plt.imshow(filtered_img, cmap='gray', interpolation='none')
        plt.title("Output {}".format(i+1))
        plt.xticks([])
        plt.yticks([])

plt.show()

