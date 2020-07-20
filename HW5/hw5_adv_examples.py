import torch
import torchvision
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torchvision.models import resnet50
from PIL import Image
import json


# fool the resnet without a target label
# the Idea is to generate some noise with gradient descent
# and add the noise to the origin image to fool the classifier
def fool_resnet(my_tensor):
    pred = model(my_tensor)
    _, max_at = pred[0].max(0) # get the true prediction label, in this case is 101
    true_class = img_class[str(max_at.item())][1]
    # here we make a trainable noise with a same size as the origin image
    noise = torch.zeros_like(my_tensor, requires_grad=True)
    optimizer = optim.SGD([noise], lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    epsilon = 1e-2 # set an epsilon to bound the noise
    for i in range(100):
        optimizer.zero_grad()
        pred = model(my_tensor + noise) # make a fake image by adding noise to the origin image and get prediction
        # loss is negative, because we want the prediction result get away from the correct prediction
        # in this case, the correct prediction is 101, so with a negative loss and an opposite gradient
        # we can get away from the result 101
        loss = -criterion(pred, torch.LongTensor([101]))
        loss.backward()
        optimizer.step()
        # bound the noise hope not to affect the final image much
        noise.data.clamp_(-epsilon, epsilon)
    # get the prediction for the fake image
    _, max_at = pred[0].max(0)
    fake_class = img_class[str(max_at.item())][1]
    print('True prediction: %s, fooled prediction: %s' %(true_class, fake_class))
    # draw plots
    plt.subplot(121)
    plt.imshow(my_tensor[0].numpy().transpose(1, 2, 0))
    plt.title('origin image')
    plt.xlabel('prediction: ' + true_class)
    plt.subplot(122)
    plt.title('fake image')
    plt.imshow((my_tensor + noise)[0].detach().numpy().transpose(1, 2, 0))
    plt.xlabel('prediction: ' + fake_class)
    plt.show()


# fool the resnet with a target label, target_num refers to the target label in the image classes
# most code are same as the above function
def fool_resnet_target(my_tensor, target_num):
    pred = model(my_tensor)
    _, max_at = pred[0].max(0)
    true_class = img_class[str(max_at.item())][1]
    noise = torch.zeros_like(my_tensor, requires_grad=True)
    optimizer = optim.SGD([noise], lr=2e-2)
    criterion = nn.CrossEntropyLoss()
    epsilon = 1e-2
    for i in range(100):
        optimizer.zero_grad()
        pred = model(my_tensor + noise)
        # everything is same with the function above, except the loss here
        # we want to fool the classifier to label the tusker as a bullet_train
        # we add a positive loss of the bullet_train, with the gradient descent
        # the classifier is more possible to predict the target label instead of the original one
        loss = -criterion(pred, torch.LongTensor([101])) + criterion(pred, torch.LongTensor([target_num]))
        loss.backward()
        optimizer.step()
        noise.data.clamp_(-epsilon, epsilon)

    _, max_at = pred[0].max(0)
    fake_class = img_class[str(max_at.item())][1]
    print('True prediction: %s, fooled prediction: %s' %(true_class, fake_class))
    plt.subplot(121)
    plt.imshow(my_tensor[0].numpy().transpose(1, 2, 0))
    plt.title('origin image')
    plt.xlabel('prediction: ' + true_class)
    plt.subplot(122)
    plt.title('fake image')
    plt.imshow((my_tensor + noise)[0].detach().numpy().transpose(1, 2, 0))
    plt.xlabel('prediction: ' + fake_class)
    plt.show()


if __name__ == '__main__':
    # read the json file and load it
    with open('imagenet_class_index.json', 'r') as json_file:
        data = json_file.read()
    img_class = json.loads(data)
    model = resnet50(pretrained=True)
    model.eval()
    # transform the image
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        # transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    # here I used Image in PIL, I did't find a similar image reading function in torchvision
    my_img = Image.open('Elephant2.jpg')
    my_tensor = preprocess(my_img)[None, :, :, :]
    fool_resnet(my_tensor)
    fool_resnet_target(my_tensor, 466)
    # plt.imshow(my_tensor[0].numpy().transpose(1, 2, 0))
    # plt.show()