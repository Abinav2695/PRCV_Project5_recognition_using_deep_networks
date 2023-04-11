#!/usr/bin/env python3

'''
 Abinav Anantharaman and Satwik Bhandiwad
   CS 5330 Spring 2023
   Recognition using Deep Networks 
'''
import torch
import torchvision
from torch import nn, optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# import utils

batch_size_test = 64
learning_rate = 0.01
momentum = 0.5
log_interval = 10

# class to run all the training and testing methods
class FashionTrain:
    def __init__(self, num_of_conv, conv_filter_size, dropout_rate):
        self.num_epochs = 5
        self.batch_size_train = 64
        self.num_of_conv = num_of_conv
        self.conv_filter_size = conv_filter_size
        self.dropout_rate = dropout_rate
        self.filename = f'curve/{self.num_epochs}_{self.batch_size_train}_{self.num_of_conv}_{self.conv_filter_size}_{self.dropout_rate}.png'
    
    #load the training and validation dataset
    def load_data(self):
        train_loader = DataLoader(
            torchvision.datasets.FashionMNIST('fashion_data', train=True, download=True,
                                    transform=torchvision.transforms.Compose([
                                        torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Normalize(
                                            (0.1307,), (0.3081,))
                                    ])),
            batch_size=self.batch_size_train)

        test_loader = DataLoader(
            torchvision.datasets.FashionMNIST('fashion_data', train=False, download=True,
                                    transform=torchvision.transforms.Compose([
                                        torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Normalize(
                                            (0.1307,), (0.3081,))
                                    ])),
            batch_size=batch_size_test)

        return train_loader, test_loader
    

    def experiment(self):
        train_loader, test_loader = self.load_data()

        network = Network(self.num_of_conv, self.conv_filter_size, self.dropout_rate)
        optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                            momentum=momentum)
        train_losses = []
        train_counter = []
        test_losses = []
        test_counter = [i * len(train_loader.dataset) for i in range(self.num_epochs + 1)]

        self.test(network, test_loader, test_losses)
        for epoch in range(1, self.num_epochs + 1):
            self.train(epoch, network, optimizer, train_loader, train_losses, train_counter)
            self.test(network,test_loader, test_losses)
        self.plot_curve(train_counter, train_losses, test_counter, test_losses)

    #train function
    def train(self,epoch, model, optimizer, train_loader, train_losses, train_counter):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))
                train_losses.append(loss.item())
                train_counter.append(
                    (batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))
                torch.save(model.state_dict(), 'results/model.pth')
                torch.save(optimizer.state_dict(), 'results/optimizer.pth')

    #test function
    def test(self,model, test_loader, test_losses):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    
    #plotting function
    def plot_curve(self, train_counter, train_losses, test_counter, test_losses):
        print(len(train_counter), len(train_counter), len(test_losses), len(test_counter))
        plt.plot(train_counter, train_losses, color='blue')
        plt.scatter(test_counter, test_losses, color='red')
        plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
        plt.xlabel('number of training examples seen')
        plt.ylabel('negative log likelihood loss')
        plt.savefig(self.filename)

    #Function to call for each variation
    def run(self):
        self.experiment()
        print('______________________________')
        print(f'Number of Epochs: {self.num_epochs}')
        print(f'Train Batch Size: {self.batch_size_train}')
        print(f'Number of Convolution Layer: {self.num_of_conv}')
        print(f'Convolution Filter Size: {self.conv_filter_size}')
        print(f'Dropout Rate: {self.dropout_rate}')
        print('______________________________')

# Network class
class Network(nn.Module):

    def __init__(self, num_of_conv, conv_filter_size, dropout_rate):
        super().__init__()
        self.input_size = 28
        self.num_of_conv = num_of_conv
        self.conv1 = nn.Conv2d(1, 10, kernel_size=conv_filter_size, padding='same')
        self.conv2 = nn.Conv2d(10, 20, kernel_size=conv_filter_size, padding='same')
        self.conv = nn.Conv2d(20, 20, kernel_size=conv_filter_size, padding='same')
        self.conv2_drop = nn.Dropout2d(dropout_rate)
        self.fc1 = nn.Linear(self.get_fc1_input_size(), 50)
        self.fc2 = nn.Linear(50, 10)

    def get_fc1_input_size(self):
        fc1_input_size = self.input_size / 2
        fc1_input_size = fc1_input_size / 2
        fc1_input_size = fc1_input_size * fc1_input_size * 20
        return int(fc1_input_size)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        for i in range(self.num_of_conv):
            x = F.relu(self.conv(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, 1)

    
# Class to run all variations
class Runner:
    def __init__(self):
        self.experiments = []

    def generate_experiments(self):
        for num_epochs in [3, 5]:
            for batch_size_train in [64, 128]:
                for num_of_conv in range(1, 4):
                    for conv_filter_size in [3, 5, 7]:
                        for dropout_rate in [0.3, 0.5]:
                            self.experiments.append(FashionTrain(num_of_conv, conv_filter_size, dropout_rate))

    def run_experiments(self):
        for experiment in self.experiments:
            experiment.run()


if __name__ == "__main__":
    runner = Runner()
    runner.generate_experiments()
    runner.run_experiments()