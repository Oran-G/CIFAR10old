print("importing torch")
import torch
import torchvision
import torchvision.transforms as transforms
import sys

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
print("importing files")
loader = torch.utils.data.DataLoader(torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform), 
                                        batch_size=16, shuffle=True, num_workers=0)
print(len(loader))   

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
trainloader = []
valloader = []
for i, data in enumerate(loader, 0):
    if i <= 2800:
        trainloader.append(data)
    else:
        valloader.append(data)    

import torch.nn as nn
import torch.nn.functional as F


from net import Net
net = Net()


print("define optims")
ler = 0.02
import torch.optim as optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.0025, momentum=0.9)
# optimizer = torch.optim.Adam(net.parameters(), lr=0.0025, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
print("begin training")
import matplotlib.pyplot as plt
x = []
y = []
valx = []
valy = []




best = 15
bestval = 15


for epoch in range(16):  # loop over the dataset multiple times
    # if epoch < 8:
    #     ler = ler/(epoch + 1)
    #     optimizer = optim.SGD(net.parameters(), lr=ler, momentum=0.9)
    correct = 0
    total = 0
    running_loss = 0.0
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
        if i % 500 == 499:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 500))
                x.append((epoch * len(trainloader)) + i)
                y.append(running_loss/500)
                PATH = './cifar_net.pth'
                torch.save(net.state_dict(), PATH)  
                
                if ((running_loss/500) - best) > (running_loss /500) / (epoch + 1):
                    running_loss = 0
                    import matplotlib.pyplot as plt
                    plt.plot(x, y, label = "train")
                    plt.plot(valx, valy, label = "valid")
                    plt.legend()
                    plt.ylabel('Running Loss')
                    plt.xlabel('Updates')
                     
                    plt.show()
                    sys.exit()
                    
                     
                best = min(best, running_loss / 500)
                running_loss = 0

                print('Accuracy of the network on the 500 update: %d %%' % (
                    100 * correct / total))
                correct = 0
                total = 0
    running_loss = 0
    for i, data in enumerate(valloader, 0):
        inputs, labels = data
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        running_loss += loss.item()
    valx.append(((epoch + 1) * len(trainloader)))
    valy.append(running_loss/len(valloader))
    bestval = min(bestval, running_loss / len(valloader))
    if running_loss/len(valloader) - bestval > (running_loss/len(valloader)) / (epoch + 1):
        import matplotlib.pyplot as plt
        plt.plot(x, y, label = "train")
        plt.plot(valx, valy, label = "valid")
        plt.legend()
        plt.ylabel('Running Loss')
        plt.xlabel('Updates')
                         
        plt.show()
        sys.exit()
    running_loss = 0






print('Finished Training')
import matplotlib.pyplot as plt
plt.plot(x, y, label = "train")
plt.plot(valx, valy, label = "valid")
plt.legend()
plt.ylabel('Running Loss')
plt.xlabel('Updates')
plt.show()
print('Finished Training')
PATH = './mnist_net.pth'
torch.save(net.state_dict(), PATH)



# for epoch in range(16):  # loop over the dataset multiple times
#     # if epoch < 8:
#     #     ler = ler/(epoch + 1)
#     #     optimizer = optim.SGD(net.parameters(), lr=ler, momentum=0.9)
#     correct = 0
#     total = 0
#     running_loss = 0.0
#     for i, data in enumerate(trainloader, 0):
#         inputs, labels = data

#         # zero the parameter gradients
#         optimizer.zero_grad()

#         # forward + backward + optimize
#         outputs = net(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#         # print statistics
#         running_loss += loss.item()
#         if i % 500 == 499:    # print every 2000 mini-batches
#             print('[%d, %5d] loss: %.3f' %
#                   (epoch + 1, i + 1, running_loss / 500))
#             x.append((epoch * len(trainloader)) + i)   
#             y.append(running_loss/500)     
#             running_loss = 0

#             print('Accuracy of the network on the 500 update: %d %%' % (
#                 100 * correct / total))
#             correct = 0
#             total = 0
#     running_loss = 0
#     for i, data in enumerate(valloader, 0):
#         inputs, labels = data
#         outputs = net(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         running_loss += loss.item()
#     valx.append(((epoch + 1) * len(trainloader)))   
#     valy.append(running_loss/len(valloader))
#     running_loss = 0     






# print('Finished Training')
# import matplotlib.pyplot as plt
# plt.plot(x, y, label = "train")
# plt.plot(valx, valy, label = "valid")
# plt.legend()
# plt.ylabel('Running Loss')
# plt.xlabel('Updates')
# plt.show()
# PATH = './cifar_net.pth'
# torch.save(net.state_dict(), PATH)