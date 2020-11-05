print("importing torch")
import torch
import torchvision
import torchvision.transforms as transforms
import sys
import torch.nn as nn
import torch.nn.functional as F
from net import Net
from datetime import datetime
import matplotlib.pyplot as plt

import torch.optim as optim
import os

def train(batch, learning_rate, epochs, batch_per):
    #create a folder for graohs
    folder = str(datetime.now())
    os.mkdir(f'./graphs/{folder}')

    #get, prepare, and split data
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    print("importing files")
    loader = torch.utils.data.DataLoader(torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform), 
                                            batch_size=batch, shuffle=True, num_workers=0)
    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    trainloader = []
    valloader = []
    for i, data in enumerate(loader, 0):
        if i <= (len(loader) / 10) * 8.5:
            trainloader.append(data)
        else:
            valloader.append(data)    

    net = Net()

    #def optim, loss, and init graph data
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    print("begin training")
    x = []
    y = []
    valx = []
    valy = []
    corx = []
    corvalx = []
    cory = []
    corvaly = []



    #these go down, and random loss is ~2.303 so 15 will be replaced
    best = 15
    bestval = 15


    for epoch in range(epochs):  # loop over the dataset multiple times
        ## possibly change lr
        # if epoch < 8:
        #     ler = ler/(epoch + 1)
        #     optimizer = optim.SGD(net.parameters(), lr=ler, momentum=0.9)
        correct = 0
        total = 0
        running_loss = 0.0
        # train mode
        net.train()
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # print statistics
            running_loss += loss.item()
            if i % batch_per == batch_per - 1:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / batch_per))
                    x.append((epoch * len(trainloader)) + i)
                    y.append(running_loss/batch_per)
                    # PATH = f'./{folder}/net.pth'
                    # torch.save(net.state_dict(), PATH)
                
                    
              
                    #possibly exit if loss goes up to much
                    # if ((running_loss/(len(trainloader) / update_checks)) - best) > (running_loss /(len(trainloader) / update_checks)) / (epoch + 1):
                    #     running_loss = 0
                    #     plt.plot(x, y, label = "train")
                    #     plt.plot(valx, valy, label = "valid")
                    #     plt.legend()
                    #     plt.ylabel('Running Loss')
                    #     plt.xlabel('Updates')
                        
                    #     plt.savefig(f'./graphs/{folder}/loss.png')
                    #     return 0
                        
                        
                    best = min(best, running_loss / batch_per)
                    running_loss = 0

                    print('Accuracy of the network on the ' + str(batch_per) + 'th update: %d %%' % (
                        100 * correct / total))
                    cory.append(100 * correct / total)   
                    corx.append((epoch * len(trainloader)) + i) 
                    correct = 0
                    total = 0
        running_loss = 0
        net.eval()
        correct = 0
        total = 0
        #check val set
        for i, data in enumerate(valloader, 0):
            inputs, labels = data
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            # total += labels.size(0)
            # correct += (predicted == labels).sum().item()
            running_loss += loss.item()
            
        valx.append(((epoch + 1) * len(trainloader)))
        valy.append(running_loss/len(valloader))
        bestval = min(bestval, running_loss / len(valloader))
        # corvaly.append(100 * correct / total)
        # corvalx.append(((epoch + 1) * len(trainloader)))
        # if val loss goes up to much exit and plot
        if running_loss/len(valloader) - bestval > (running_loss/len(valloader)) / ((epoch + 1) * 1):
            
            plt.plot(x, y, label = "train")
            plt.plot(valx, valy, label = "valid")
            plt.legend()
            plt.ylabel('Running Loss')
            plt.xlabel('Updates')
            plt.savefig(f'./graphs/{folder}/loss.png')  
            plt.clf()   

            plt.plot(corx, cory, label = "train")
            # plt.plot(corvalx, corvaly, label = "valid")
            plt.legend()
            plt.ylabel('Accuracy')
            plt.xlabel('Updates')
            plt.savefig(f'./graphs/{folder}/accuracy.png')                      
            return [plt, net]
        running_loss = 0
        correct = 0
        total = 0
        





    # finish training. in future dont plot and save here just return them
    print('Finished Training')
    plt.plot(x, y, label = "train")
    plt.plot(valx, valy, label = "valid")
    plt.legend()
    plt.ylabel('Running Loss')
    plt.xlabel('Updates')
    plt.savefig(f'./graphs/{folder}/loss.png')
    plt.clf()  
    plt.plot(corx, cory, label = "train")
    plt.plot(corvalx, corvaly, label = "valid")
    plt.legend()
    plt.ylabel('Accuracy')
    plt.xlabel('Updates')
    plt.savefig(f'./graphs/{folder}/accuracy.png')    
    return [plt, net]
    # PATH = f'./{folder}/net.pth'
    # torch.save(net.state_dict(), PATH)

