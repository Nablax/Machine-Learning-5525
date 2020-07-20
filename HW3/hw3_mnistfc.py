import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt


# draw one line
def plt_draw(data, data_name, x_label, y_label):
    plt.plot(data, label=data_name)
    plt.xticks(range(len(data)))
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.show()


# model here
class multi_layer_fc(nn.Module):
    def __init__(self):
        super(multi_layer_fc, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x


# model training
def train_model(trainset):
    net = multi_layer_fc()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=32, shuffle=True)
    epoch_loss = []
    epoch_trn_acc = []
    for epoch in range(10):
        loss_per_epoch = 0
        for i, (data, labels) in enumerate(trainloader, 0):
            data = data.view(-1, 28 * 28)
            optimizer.zero_grad()
            outputs = net(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_per_epoch += loss.item() * len(labels)
        epoch_loss.append(loss_per_epoch / len(trainset))
        epoch_trn_acc.append(test_model(trainset, net))
    plt_draw(epoch_loss, 'loss per epoch', 'epoch', 'loss')
    plt_draw(epoch_trn_acc, 'train accuracy per epoch', 'epoch', 'accuracy')
    return net


# model testing
def test_model(testset, net):
    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=32, shuffle=True)
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            data = images.view(-1, 28 * 28)
            outputs = net(data)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy: %f %%' % (100 * correct / total))
    acc = 100 * correct / total
    return acc


if __name__ == '__main__':
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
    weight_path = './mnist-fc'

    # save model
    # net = train_model(trainset)
    # torch.save(net.state_dict(), weight_path)

    # read model
    net = multi_layer_fc()
    net.load_state_dict(torch.load(weight_path))
    test_model(testset, net)


