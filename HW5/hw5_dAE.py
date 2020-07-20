import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# draw one line
def plt_draw(data, data_name, x_label, y_label):
    plt.plot(data, label=data_name)
    plt.xticks(range(len(data)))
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.show()


# draw multiple lines once
def plt_draw_lines(data, data_label, x_label, y_label):
    for i in range(len(data_label)):
        if isinstance(data_label[i], int):
            plt.plot(data[i], label='batch size %d' %(data_label[i]))
        else:
            plt.plot(data[i], label=data_label[i])
    plt.xticks(range(len(data[0])))
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.show()


# model here
class my_dAE(nn.Module):
    def __init__(self):
        super(my_dAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 400),
            nn.ReLU(),
            nn.Linear(400, 20),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(20, 400),
            nn.ReLU(),
            nn.Linear(400, 784),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# model training, opt is optimizer
def train_model(trainset, batch_size=100, epochs=50):
    dAE = my_dAE()
    criterion = nn.BCELoss()
    optim_dae = optim.Adam(dAE.parameters(), lr=0.001)
    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)
    epoch_loss = []
    for epoch in range(epochs):
        loss_value = 0
        print('Start training in epoch %d' %(epoch))
        for i, (data, _) in enumerate(trainloader, 0):
            # train the dAE
            data = data.view(-1, 784)
            noise = torch.randn((data.size(0), 784))
            noise_data = data + noise
            optim_dae.zero_grad()
            d_out = dAE(noise_data)
            loss = criterion(d_out, data)

            loss_value += loss.item() * data.size(0) # add loss in every batch
            loss.backward()
            optim_dae.step()

        gen_picture(dAE, trainset, pic_num=5)
        epoch_loss.append(loss_value / len(trainset)) # get average loss perepoch
        print('loss: %f'%(epoch_loss[epoch]))
    return epoch_loss, dAE


# generate denoised pictures and compare with the origin pictures
def gen_picture(dAE, data_set, pic_num=5):
    one_batch = next(iter(torch.utils.data.DataLoader(dataset=data_set, batch_size=pic_num, shuffle=True)))[0]
    row = 2
    col = pic_num
    noise = torch.randn((pic_num, 784))
    noise_data = one_batch.view(pic_num, -1) + noise
    d_noise_data = dAE(noise_data).detach()
    for i in range(pic_num):
        # draw noise data
        plt.subplot(row, col, i + 1)
        plt.xticks([])
        plt.yticks([])
        sub_pic = noise_data[i].view(28, 28).numpy()
        plt.imshow(sub_pic, cmap='gray')
        # draw denoised data in the same column
        plt.subplot(row, col, i + 1 + pic_num)
        plt.xticks([])
        plt.yticks([])
        sub_pic = d_noise_data[i].view(28, 28).numpy()
        plt.imshow(sub_pic, cmap='gray')
    plt.show()

if __name__ == '__main__':
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    net_pth = './hw5_dAE.pth'
    # train the model
    epoch_loss, dAE = train_model(trainset, 64, 10)
    gen_picture(dAE, trainset, 5)
    torch.save(dAE.state_dict(), net_pth)
    plt_draw(epoch_loss, 'dAE loss', 'epoch', 'loss')
    # load and test model
    dAE = my_dAE()
    dAE.load_state_dict(torch.load(net_pth))
    gen_picture(dAE, trainset)
