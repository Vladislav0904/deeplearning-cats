import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.utils
from torch import load
from torch.nn import Module, Conv2d, Linear, MaxPool2d, AdaptiveMaxPool1d
from torch.nn.functional import relu, dropout

from imageLoader import *


# functions to show an image


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


num_epochs = 2
PATH = 'model/catvsdogs.pth'


class Network(Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv_1 = Conv2d(in_channels=3, out_channels=64, kernel_size=5)
        self.conv_2 = Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.conv_3 = Conv2d(in_channels=128, out_channels=256, kernel_size=5)

        self.maxPooling = MaxPool2d(kernel_size=4)
        self.adPooling = AdaptiveMaxPool1d(256)

        self.fc1 = Linear(in_features=256, out_features=128)
        self.fc2 = Linear(in_features=128, out_features=64)
        self.out = Linear(in_features=64, out_features=2)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.maxPooling(x)
        x = relu(x)

        x = self.conv_2(x)
        x = self.maxPooling(x)
        x = relu(x)

        x = self.conv_3(x)
        x = self.maxPooling(x)
        x = relu(x)

        x = dropout(x)
        x = x.view(1, x.size()[0], -1)
        x = self.adPooling(x).squeeze()

        x = self.fc1(x)
        x = relu(x)

        x = self.fc2(x)
        x = relu(x)

        return relu(self.out(x))


#
# for iter in range(4, 50):
#     network = Network()
#     criterion = CrossEntropyLoss()
#     optimizer = optim.SGD(network.parameters(), lr=0.001, momentum=0.9)
#     network.load_state_dict(torch.load('model/catvsdogs.pth'))
#     imageLoader = ImageLoader(trainData, transform)
#     dataLoader = DataLoader(imageLoader, batch_size=10, shuffle=True)
#
#     for epoch in range(num_epochs):
#         running_loss = 0.0
#         for i, datas in enumerate(dataLoader, iter * 500):
#             inputs, labels = datas
#             optimizer.zero_grad()
#             outputs = network(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item()
#             if i % 500 == 499:
#                 print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 500:.3f}')
#                 running_loss = 0.0
#                 break
#
#     print("Finished iteration " + str(iter))
#     save(network.state_dict(), PATH)
#     torch.cuda.empty_cache()


imageLoader = ImageLoader(trainData, transform)
dataLoader = DataLoader(imageLoader, batch_size=4, shuffle=True)
dataiter = iter(dataLoader)
images, labels = next(dataiter)

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join(f'{labels[j]}' for j in range(4)))
network = Network()
network.load_state_dict(load(PATH))
outputs = network(images)
_, predicted = torch.max(outputs, 1)
print('Predicted: ', ' '.join(f'{predicted[j]}'
                              for j in range(4)))