# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 00:00:39 2021

@author: mason
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


XComplete = np.load('datArr.npy')
YComplete = np.load('targArr.npy')
XComplete = np.delete(XComplete, 0, axis = 3)
XComplete = np.swapaxes(XComplete, 3, 0)
XComplete = np.swapaxes(XComplete, 1, 2)
YComplete = np.delete(YComplete, 0, axis = 0)
yY = np.copy(YComplete)
test = np.copy(XComplete[1,:,:,:])
test2 = np.copy(XComplete[0,:,:,:])
#test = np.squeeze(test, axis=1)

##  ----------------------------------------------------
##  Create the NN model with a double convolutional step
##  ----------------------------------------------------

##
##  Convert to tensors
##
XComplete = torch.from_numpy(XComplete).double()
YComplete = torch.from_numpy(YComplete).double()



##
##  Define the neural net
##
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(2, 5, 7).double() #two input layers, five output layers, 7x7 kernel
        self.pool = nn.MaxPool2d(2, 2).double() #maxpool
        self.conv2 = nn.Conv2d(5, 8, 3).double()    
        self.fc1 = nn.Linear(20*20*8, 120).double()
        self.fc2 = nn.Linear(120, 10).double()
        self.fc3 = nn.Linear(10, 2).double()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 20*20*8)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
model = Net()
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.0000001, momentum=0.5)

##
## Train the neural net
##
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i in range(700):
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        yPred = model(XComplete)
        testY = yPred.detach().numpy()
        loss = loss_fn(yPred, YComplete)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2 == 0:    # print every 2 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss))
            running_loss = 0.0

print('Finished Training')
k = model(XComplete)
yVal = k.detach().numpy()