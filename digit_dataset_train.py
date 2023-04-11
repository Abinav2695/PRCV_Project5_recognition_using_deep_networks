#! /usr/bin/env python3
'''
 Abinav Anantharaman and Satwik Bhandiwad
   CS 5330 Spring 2023
   Recognition using Deep Networks 
'''

import torch
import torchvision
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#training parameters
n_epochs = 15
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10

train_losses = []
train_counter = []
test_losses = []
test_counter = []

random_seed = 42
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

# Network class initialization
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(p=0.5) #added
        self.flatten = nn.Flatten()  #added
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        self.train_loader = None
        self.test_loader = None

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)  #Flattening
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

    # Function to get the training and testing datasets
    def get_dataset(self, show_examples=True):

        global test_counter
        self.train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('files/', train=True, download=True,
                                    transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                        (0.1307,), (0.3081,))
                                    ])),
        batch_size=batch_size_train, shuffle=True)

        self.test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('files/', train=False, download=True,
                                    transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                        (0.1307,), (0.3081,))
                                    ])),
        batch_size=batch_size_test, shuffle=True)

        print("Train dataset pass: {} , Test dataset pass: {}".format(self.train_loader, self.test_loader))

        examples = enumerate(self.test_loader)
        batch_idx, (self.example_data, self.example_targets) = next(examples)
        print(self.example_data.shape)
        test_counter = [i*len(self.train_loader.dataset) for i in range(n_epochs + 1)]
        
        if show_examples:
            fig = plt.figure()
            for i in range(6):
                plt.subplot(2,3,i+1)
                plt.tight_layout()
                plt.imshow(self.example_data[i][0], cmap='gray', interpolation='none')
                plt.title("Ground Truth: {}".format(self.example_targets[i]))
                plt.xticks([])
                plt.yticks([])
            plt.show()

        
    # function to train the model
    def train_network(self, optimizer, epoch, model_file_name, optimizer_file_name):
        self.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            optimizer.zero_grad()
            output = self(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader), loss.item()))
                train_losses.append(loss.item())
                train_counter.append(
                    (batch_idx*64) + ((epoch-1)*len(self.train_loader.dataset)))
                torch.save(self.state_dict(), model_file_name +'.pth')
                torch.save(optimizer.state_dict(), optimizer_file_name + '.pth')

    #function to test the model
    def test_network(self):
        self.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                output = self(data)
                test_loss += F.nll_loss(output, target, size_average=False).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(self.test_loader.dataset)
        test_losses.append(test_loss)
        print("Test loss size: {}".format(len(test_losses)))
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(self.test_loader.dataset),
            100. * correct / len(self.test_loader.dataset)))
        
    #Run function to run training and testing simultaneously
    def train_and_test(self, optimizer):
        self.test_network()
        for epoch in range(1, n_epochs + 1):
            self.train_network(optimizer, epoch,'model2.pth', 'optimizer2.pth')
            self.test_network()
        
        print(len(test_counter), len(test_losses))
  
        fig = plt.figure()
        plt.plot(train_counter, train_losses, color='blue')
        plt.scatter(test_counter, test_losses, color='red')
        plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
        plt.xlabel('number of training examples seen')
        plt.ylabel('negative log likelihood loss')
        plt.show()

if __name__ == '__main__':

    print('Digit Recognition Project')
    my_net = Net()
    my_net.get_dataset()
    optimizer = optim.SGD(my_net.parameters(), lr=learning_rate,
                      momentum=momentum)
    
    my_net.train_and_test(optimizer)
    my_net.load_state_dict(torch.load('model2.pth'))


    with torch.no_grad():
        output = my_net(my_net.example_data)

    fig = plt.figure()
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.tight_layout()
        plt.imshow(my_net.example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Prediction: {}".format(
            output.data.max(1, keepdim=True)[1][i].item()))
        plt.xticks([])
        plt.yticks([])
    plt.show()



