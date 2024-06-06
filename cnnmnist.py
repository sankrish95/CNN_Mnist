import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

"""FCC network
class NN(nn.Module):
    def __init__(self,input_size,num_classes):
        super(NN,self).__init__()
        self.fc1 = nn.Linear(input_size,10)
        self.fc2 = nn.Linear(10,num_classes)

    def forward(self,x): #input size =64*784
        x = F.relu(self.fc1(x)) # 64*50
        x = self.fc2(x) #64*10
        return x
"""
#CNN model

class CNN(nn.Module):
    def __init__(self,in_channels=1,num_classes=10):
        super(CNN,self).__init__()
        self.Conv1 = nn.Conv2d(in_channels=1,out_channels=8,kernel_size=(3,3),padding=(1,1))
        self.pool =  nn.MaxPool2d(stride=(2,2),kernel_size=(2,2))
        self.Conv2 = nn.Conv2d(in_channels=8,out_channels=16,kernel_size=(3,3),padding=(1,1))
        self.fc1 = nn.Linear(16*7*7,512)
        self.fc2 = nn.Linear(512,num_classes)
    def forward(self,x):
        x = F.relu(self.Conv1(x))
        x = self.pool(x)
        x = F.relu(self.Conv2(x))
        x = self.pool(x)
        print(x.shape)
        x = x.reshape(x.shape[0], -1)
        print(x.shape)
        x = self.fc1(x)
        x = F.relu(x)
        print(x.shape)
        x = self.fc2(x)
        print(x.shape)
        return x

model = CNN()
x = torch.randn(68,1,28,28)
print(model(x).shape)




if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

#Hyperparameters

in_channels = 784
num_classes = 10
learning_rate = 0.01
batch_size = 64
num_epochs = 5

#Load data

train_dataset = datasets.MNIST(root ='/dataset',train=True,transform=transforms.ToTensor(),download=True)
train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
test_dataset = datasets.MNIST(root='/dataset',train=False,transform=transforms.ToTensor(),download=True)
test_loader = DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=True)

#init model
model = CNN().to(device)
#Loss and optim

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr=learning_rate)

#Train Network

for epoch in range(num_epochs):
    avg= 0
    for batch_idx,(data,target) in enumerate(train_loader):
        data = data.to(device=device)
        targets = target.to(device=device)

        #print(data.shape)

        #forwrd prop
        scores = model(data)
        loss = loss_fn(scores,targets)
        avg =+ loss.item()
        #backwrd prop
        optimizer.zero_grad()
        loss.backward()
        #grad descent
        optimizer.step()
        #print(loss.item())



    avg = avg/(batch_idx+1)
    #print('epoch = {}, train loss = {}'.format(epoch, avg))
    avg_test = 0
    avg_test_acc=0
    with torch.no_grad():
        for batch_idx2, (data, target) in enumerate(test_loader):
            data = data.to(device=device)
            targets = target.to(device=device)

            #print(data.shape)

            #forwrd prop

            scores = model(data)
            loss = loss_fn(scores, targets)
            avg_test = + loss.item()

            pred = torch.argmax(scores,dim=1,keepdim=False)
            #print(targets)
            #print(pred)

            z = np.array(targets) - np.array(pred)
            non_zeros = np.count_nonzero(z)
            correct = targets.shape[0] - non_zeros
            acc = correct/targets.shape[0]
            avg_test_acc += acc

    avg_test = avg_test/(batch_idx2+1)
    avg_test_acc = avg_test_acc/(batch_idx2+1)
    print('epoch = {}, test loss = {} ,test acc={}'.format(epoch, avg_test,avg_test_acc))
torch.save(model.state_dict(),f="CNN_mnist.pth")



