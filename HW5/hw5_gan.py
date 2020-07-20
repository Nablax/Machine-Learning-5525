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


# discriminator model here
class my_discriminator(nn.Module):
    def __init__(self):
        super(my_discriminator, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), negative_slope=0.2)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.2)
        x = self.sig(self.fc3(x))
        return x


# generate model here
class my_generator(nn.Module):
    def __init__(self):
        super(my_generator, self).__init__()
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 784)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), negative_slope=0.2)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.2)
        x = self.tanh(self.fc3(x))
        return x


# model training, opt is optimizer
def train_model(trainset, batch_size=100, epochs=50):
    gen = my_generator()
    dis = my_discriminator()
    criterion = nn.BCELoss()
    optim_gen = optim.Adam(gen.parameters(), lr=0.0002)
    optim_dis = optim.Adam(dis.parameters(), lr=0.0002)
    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)
    epoch_gen_loss = []
    epoch_dis_loss = []
    for epoch in range(epochs):
        gen_loss_value = dis_loss_value = 0
        print('Start training in epoch %d' %(epoch))
        for i, (data, _) in enumerate(trainloader, 0):

            data = data.view(batch_size, -1)
            z = torch.randn((batch_size, 128))
            true_label = torch.ones(batch_size, 1)
            false_label = torch.zeros(batch_size, 1)
            # train the discriminator first
            optim_dis.zero_grad()
            true_out = dis(data)
            false_out = dis(gen(z))
            dis_loss = criterion(true_out, true_label) + criterion(false_out, false_label)
            dis_loss_value += dis_loss.item() * batch_size
            dis_loss.backward()
            optim_dis.step()
            # train the generator next
            optim_gen.zero_grad()
            false_out = dis(gen(z))
            gen_loss = criterion(false_out, true_label)
            gen_loss_value += gen_loss.item() * batch_size # add total loss per batch
            gen_loss.backward()
            optim_gen.step()
        epoch_gen_loss.append(gen_loss_value / len(trainset))# find average loss
        epoch_dis_loss.append(dis_loss_value / len(trainset))
        print('gen loss: %f, dis loss: %f'%(gen_loss_value / len(trainset), dis_loss_value / len(trainset)))
        if (epoch + 1) % 10 == 0:
            gen_picture(gen)
    return gen, dis, epoch_gen_loss, epoch_dis_loss

# draw 4 by 4 grid of the generated pictures
def gen_picture(gen, pic_num=16):
    z = torch.randn((pic_num, 128))
    gen_pic = gen(z).detach()
    row = col = int(np.sqrt(pic_num))
    for i in range(pic_num):
        plt.subplot(row, col, i + 1)
        plt.xticks([])
        plt.yticks([])
        sub_pic = gen_pic[i].view(28, 28).numpy()
        plt.imshow(sub_pic, cmap='gray')
    plt.show()

if __name__ == '__main__':
    # to transform the dataset into mean 0.5, std 0.5
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    # train the model and get all loss
    gen, dis, epoch_gen_loss, epoch_dis_loss = train_model(trainset, 100, 50)
    gen_pth = './hw5_gan_gen.pth'
    dis_pth = './hw5_gan_dis.pth'
    # save the model and draw plot
    torch.save(gen.state_dict(), gen_pth)
    torch.save(dis.state_dict(), dis_pth)
    plt_draw(epoch_gen_loss, 'generator loss', 'epoch', 'loss')
    plt_draw(epoch_dis_loss, 'discriminator loss', 'epoch', 'loss')
    # test model here
    gen = my_generator()
    gen.load_state_dict(torch.load(gen_pth))
    gen_picture(gen)





