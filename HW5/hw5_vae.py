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
class my_VAE(nn.Module):
    def __init__(self):
        super(my_VAE, self).__init__()
        self.fc1 = nn.Linear(784, 400)
        self.mu = nn.Linear(400, 20)
        self.sigma = nn.Linear(400, 20)
        self.fc2 = nn.Linear(20, 400)
        self.fc3 = nn.Linear(400, 784)
        self.sig = nn.Sigmoid()

    def encode(self, x):
        x = F.relu(self.fc1(x))
        mu = self.mu(x)
        sigma = self.sigma(x)
        return mu, sigma

    def decode(self, x):
        x = F.relu(self.fc2(x))
        x = self.sig(self.fc3(x))
        return x

    def forward(self, x):
        mu, sigma = self.encode(x)
        z = mu + sigma * torch.randn_like(mu)
        x = self.decode(z)
        return x, mu, sigma


# loss function, loss = BCE + kl divergence
def loss_function(x, x_gen, mu, sigma):
    BCE_loss = nn.BCELoss(reduction='sum')
    # as kl divergence should be larger than 0, I multiple -1 in the kl divergence
    kl_divergence = -0.5 * torch.sum(1 + torch.log(sigma**2) - mu**2 - sigma**2)
    return BCE_loss(x_gen, x) + kl_divergence


# model training, optim_vae is optimizer
def train_model(trainset, batch_size=64, epochs=10):
    VAE = my_VAE()
    optim_vae = optim.Adam(VAE.parameters(), lr=0.0005)
    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)
    epoch_loss = []
    for epoch in range(epochs):
        loss_value = 0
        print('Start training in epoch %d' %(epoch))
        for i, (data, _) in enumerate(trainloader, 0):
            data = data.view(-1, 784)
            optim_vae.zero_grad()
            d_out, mu, sigma = VAE(data)
            loss = loss_function(data, d_out, mu, sigma)

            loss_value += loss.item()
            loss.backward()
            optim_vae.step()

        gen_picture(VAE, pic_num=16)
        epoch_loss.append(loss_value / len(trainset))
        print('loss: %f'%(epoch_loss[epoch]))
    return epoch_loss, VAE


# generate 16 by 16 images
def gen_picture(VAE, pic_num=16):
    z = torch.randn((pic_num, 20))
    gen_pic = VAE.decode(z).detach()
    row = col = int(np.sqrt(pic_num))
    for i in range(pic_num):
        plt.subplot(row, col, i + 1)
        plt.xticks([])
        plt.yticks([])
        sub_pic = gen_pic[i].view(28, 28).numpy()
        plt.imshow(sub_pic, cmap='gray')
    plt.show()


if __name__ == '__main__':
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    net_pth = './hw5_vae.pth'
    # train the model
    epoch_loss, VAE = train_model(trainset, 64, 10)
    gen_picture(VAE, 16)
    torch.save(VAE.state_dict(), net_pth)
    plt_draw(epoch_loss, 'VAE loss', 'epoch', 'loss')
    # load the model and test
    VAE = my_VAE()
    VAE.load_state_dict(torch.load(net_pth))
    gen_picture(VAE, 16)
