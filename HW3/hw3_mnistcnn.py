import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
# import time


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
class my_CNN(nn.Module):
    def __init__(self):
        super(my_CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(14 * 14 * 20, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.pool(self.conv1(x)))
        x = x.view(-1, 14 * 14 * 20)
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x


# model training, opt is optimizer
def train_model(trainset, batch_size, opt='SGD'):
    net = my_CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    if opt == 'ADAM':
        optimizer = optim.Adam(net.parameters(), lr=0.001)
    if opt == 'ADAGRAD':
        optimizer = optim.Adagrad(net.parameters(), lr=0.01)
    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)
    epoch_loss = []
    epoch_trn_acc = []
    # start = time.time()
    # if_converged = False
    # converge_loss = 0
    epochs = 10
    for epoch in range(epochs):
        loss_per_epoch = 0
        # if(if_converged):
        #     break
        for i, (data, labels) in enumerate(trainloader, 0):
            optimizer.zero_grad()
            outputs = net(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_per_epoch += loss.item() * len(labels)
            # if loss.item() < converge_loss:
            #     end = time.time()
            #     print('The batch size is %d, the time to converge is: %0.2f' % (batch_size, end - start))
            #     if_converged = True
            #     break
        epoch_loss.append(loss_per_epoch / len(trainset))
        epoch_trn_acc.append(test_model(trainset, net))
    # plt_draw(epoch_loss, 'loss per epoch with batch size %d' %(batch_size), 'epoch', 'loss')
    # plt_draw(epoch_trn_acc, 'train accuracy per epoch', 'epoch', 'accuracy')
    return net, epoch_loss, epoch_trn_acc


# model test
def test_model(testset, net):
    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=32, shuffle=True)
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy: %f %%' % (100 * correct / total))
    acc = 100 * correct / total
    return acc

if __name__ == '__main__':
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
    weight_path = './mnist-cnn'
    batch_size_list = [32, 64, 96, 128] # set any batch size as will
    # batch_size_list = [32]

    #train model and draw lines
    epoch_loss_list = []
    epoch_trn_acc_list = []
    for batch_size in batch_size_list:
        net, epoch_loss, epoch_trn_acc = train_model(trainset, batch_size)
        epoch_loss_list.append(epoch_loss)
        epoch_trn_acc_list.append(epoch_trn_acc)

    plt_draw_lines(epoch_loss_list, batch_size_list,'epoch', 'loss')
    plt_draw_lines(epoch_trn_acc_list, batch_size_list, 'epoch', 'accuracy')

    # save and read model
    # torch.save(net.state_dict(), weight_path)
    # net = my_CNN()
    # net.load_state_dict(torch.load(weight_path))
    # test_model(testset, net)

    # different optimizer test
    torch_optim_types = ['SGD', 'ADAM', 'ADAGRAD']
    epoch_loss_list = []
    epoch_trn_acc_list = []
    for optim_type in torch_optim_types:
        net, epoch_loss, epoch_trn_acc = train_model(trainset, 32, optim_type)
        epoch_loss_list.append(epoch_loss)
        epoch_trn_acc_list.append(epoch_trn_acc)
    plt_draw_lines(epoch_loss_list, torch_optim_types, 'epoch', 'loss')
    plt_draw_lines(epoch_trn_acc_list, torch_optim_types, 'epoch', 'accuracy')




