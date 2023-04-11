#!/usr/bin/env python3

'''
 Abinav Anantharaman and Satwik Bhandiwad
   CS 5330 Spring 2023
   Recognition using Deep Networks 
'''

import cv2
import torch
import torchvision
import numpy as np
from torchsummary import summary
import matplotlib.pyplot as plt
import matplotlib

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os
import sys
from PIL import Image

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from digit_dataset_train import Net



# greek data set transform
class GreekTransform:
    def __init__(self):
        pass

    # def __call__(self, x):
    #     x = torchvision.transforms.functional.rgb_to_grayscale( x )
    #     x = torchvision.transforms.functional.affine( x, 0, (0,0), 36/128, 0 )
    #     x = torchvision.transforms.functional.center_crop( x, (28, 28) )
    #     return torchvision.transforms.functional.invert( x )

    def __call__(self, x):
        if type(x) == Image.Image: 
            width, height = x.size
        elif torch.is_tensor(x):
            width, height = x.shape[1], x.shape[2]
        else:    
            raise TypeError("Input must be a PIL image or Tensor image")
        
        x = torchvision.transforms.functional.rgb_to_grayscale( x )
        x = torchvision.transforms.functional.affine(x, 0, (0,0), 33/min(width, height), 0)
        # x = torchvision.transforms.functional.affine(x, 0, (0,0), 36/128, 0)
        x = torchvision.transforms.functional.center_crop( x, (28, 28) )
        return torchvision.transforms.functional.invert( x )

# Greek Network class with modification of MINST network class
class GreekNet(nn.Module):
    """
    A PyTorch neural network that classifies Greek letters.
    """
    def __init__(self, out_features=3):
        super(GreekNet, self).__init__()
        
        # initialize a MyNetwork instance to classify digits
        self.digit_network = Net()
        
        # remove the last fully connected layer from the MNIST network
        self.digit_network.fc2 = nn.Identity()
        
        # add a fully connected layer to classify Greek letters
        self.fc = nn.Linear(in_features=50, out_features=out_features)

    def forward(self, x):
        # pass input image through the MNIST network
        x = self.digit_network(x)
        
        # pass output of the MNIST network through the added fully connected layer
        x =  self.fc(x)
        
        # apply log_softmax activation to the output of the fully connected layer
        return F.log_softmax(x)


if __name__=='__main__':

    """
    Main function to call the necessary functions. It sets up the initial parameters and
    calls the training.
    """
     # Set hyperparameters
    learning_rate = 0.01
    momentum = 0.5
    n_epochs = 35

    # Set random seed
    random_seed = 42
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    # Initialize the GreekNet model and load the pretrained weights from the MNIST task
    greek_model = GreekNet(out_features=3)
    greek_model.digit_network.load_state_dict(torch.load('model_satwik.pth'), strict=False)

    # Freeze the weights of the MNIST layers so they are not updated during training on the Greek data set
    for param in greek_model.digit_network.parameters():
        param.requires_grad = False

    # Set the optimizer for training
    optimizer = optim.SGD(greek_model.parameters(), lr=learning_rate, momentum=momentum)

    # Print a summary of the model architecture
    print(summary(greek_model, (1, 28, 28), batch_size=1))

    # Load the training data for the Greek data set
    greek_train = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder('image_dataset/greek_train/',
                                         transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                                                   GreekTransform(),
                                                                                   torchvision.transforms.Normalize(
                                                                                       (0.1307,), (0.3081,))])),
                                               batch_size=5,
                                               shuffle=True)

    # Save a sample of the training data
    idx, (sampleX, sampleY) = next(enumerate(greek_train))
    figure = plt.figure()
    for i in range(5):
        plt.subplot(2,3,i+1) # create a subplot of 2 rows and 3 columns, and select the i-th subplot
        plt.tight_layout() # automatically adjust subplot parameters to fit the figure area
        plt.imshow(sampleX[i][0], cmap='gray', interpolation='none') # display the image in grayscale
        plt.title("Actual Label: {}".format(sampleY[i])) # set the title of the subplot with the actual label of the image
        plt.xticks([]) # remove the x-axis ticks
        plt.yticks([]) # remove the y-axis ticks
    figure.savefig('sample_data_greek.png') # save the figure as a png file
    train_losses = []
    train_counter = []
    log_interval = 10

    def train(epoch):
        """
        This method trains the above defined network. The function also
        plots few images from the dataset for visualization
        """
        # Set the model to train mode
        greek_model.train()
        for batch_idx, (data, target) in enumerate(greek_train):
            optimizer.zero_grad()
            output = greek_model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            # Print the progress and append the loss and counter to their respective lists
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(greek_train.dataset),
                    100. * batch_idx / len(greek_train), loss.item()))
                train_losses.append(loss.item())
                train_counter.append(
                    (batch_idx*64) + ((epoch-1)*len(greek_train.dataset)))
                # Task 1E
                # Save the model and optimizer state to disk
                torch.save(greek_model.state_dict(), 'greek-model.pth')
                torch.save(optimizer.state_dict(), 'greek-optimizer.pth')
    
    # Loop over the specified number of epochs
    for epoch in range(1, n_epochs + 1):
        train(epoch)

    # Create a figure for the train and test loss plot
    fig = plt.figure()
    plt.plot(train_counter, train_losses, color='blue')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    fig.savefig('train_test_loss_greek.png')


    # load Greek dataset and set model to evaluation mode
    greek_train = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder( 'image_dataset/greek_train/',
                                          transform = torchvision.transforms.Compose( [torchvision.transforms.ToTensor(),
                                                    GreekTransform(),
                                                    torchvision.transforms.Normalize(
                                                    (0.1307,), (0.3081,) ) ] ) ),
        batch_size = 5,
        shuffle = True )
    greek_model.eval()
    # initialize empty lists to store data, predicted labels, and correct labels
    datas, predicts, targets = [],[],[]
    # make predictions on Greek dataset
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(greek_train):
            print(batch_idx, target)
            pred = greek_model(data)
            predicted = torch.max(pred.data, 1)[1]
            for j in range(len(target)):
                datas.append(data[j][0])
                predicts.append(predicted[j])
                targets.append(target[j])
                output_values = ["%.2f" % x for x in pred[j].tolist()]
                print(f"Output values for example {j+1}: {output_values}")
                print(f"Index of the max output value: {np.argmax(output_values)}")
                print(f"Predicted label: {predicted[j]}")
                print(f"Correct label: {target[j]}")
    # plot predicted data with labels
    matplotlib.rcParams.update({'font.size': 10})

    fig, axes = plt.subplots(6, 6, sharex=True, sharey=True)
    plt.tight_layout()
    for i,ax in enumerate(axes.flat):
        if i<len(datas): 
            ax.imshow(datas[i], cmap='gray', interpolation='none')
            ax.set_axis_on()
            ax.set(xticks=[],yticks=[])
            ax.set(title="P: {} A: {}".format(predicts[i],targets[i]))
        else:
            ax.set_axis_off()
    fig.savefig('predicted_data_greek.png')


    # Load handwritten greek dataset
    greek_test = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder( 'image_dataset/greek_test/',
                                          transform = torchvision.transforms.Compose( [torchvision.transforms.ToTensor(),
                                                    GreekTransform(),
                                                    torchvision.transforms.Normalize(
                                                    (0.1307,), (0.3081,) ) ] ) ),
        batch_size = 5,
        shuffle = True )
    greek_model.eval()
    datas, predicts, targets = [],[],[]
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(greek_test):
            print(batch_idx, target)
            pred = greek_model(data)
            predicted = torch.max(pred.data, 1)[1]
            for j in range(len(target)):
                datas.append(data[j][0])
                predicts.append(predicted[j])
                targets.append(target[j])
                output_values = ["%.2f" % x for x in pred[j].tolist()]
                print(f"Output values for example {j+1}: {output_values}")
                print(f"Index of the max output value: {np.argmax(output_values)}")
                print(f"Predicted label: {predicted[j]}")
                print(f"Correct label: {target[j]}")

    matplotlib.rcParams.update({'font.size': 10})

    fig, axes = plt.subplots(3, 3, sharex=True, sharey=True)
    plt.tight_layout()
    for i,ax in enumerate(axes.flat):
        if i<=8: 
            ax.imshow(datas[i], cmap='gray', interpolation='none')
            ax.set_axis_on()
            ax.set(xticks=[],yticks=[])
            ax.set(title="P: {} A: {}".format(predicts[i],targets[i]))
        else:
            ax.set_axis_off()
    fig.savefig('predicted_data_greek_handwritten.png')



    



    

