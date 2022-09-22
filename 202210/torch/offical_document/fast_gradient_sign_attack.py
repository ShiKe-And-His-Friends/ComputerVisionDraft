import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets ,transforms
import numpy as np
import matplotlib.pyplot as plt

use_cuda = False
epsilons = [0 ,0.05 ,.1 ,.15 ,.2 ,.25 ,.3]
pretrained_model = "../weight/lenet_mnist_model.pth"

class Net(nn.Module):
    def __init__(self):
        super(Net ,self).__init__()
        self.conv1 = nn.Conv2d(1 ,10 ,kernel_size=5)
        self.conv2 = nn.Conv2d(10 ,20 ,kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320 ,50)
        self.fc2 = nn.Linear(50 ,10)

    def forward(self ,x):
        x = F.relu(F.max_pool2d(self.conv1(x) ,2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)) ,2))
        x = x.view(-1 ,320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x ,training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x ,dim=1)

def FastGradient():
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data' ,train=False ,download=True ,
                       transform=transforms.Compose([transforms.ToTensor(),])
                       ),
        batch_size = 1,
        shuffle = True
    )
    print("CUDA Available:" ,torch.cuda.is_available())
    device = torch.device("cuda" if (use_cuda and torch.cude.is_availabel()) else "cpu")
    # initialze network
    model = Net().to(device)
    # upload fine-tune model
    model.load_state_dict(torch.load(pretrained_model ,map_location='cpu'))
    # evaluate model
    eval_result = model.eval()
    print(eval_result)

    # execuate distribution
    accuracies = []
    examples = []
    for eps in epsilons:
        acc ,ex = test(model ,device ,test_loader ,eps)
        accuracies.append(acc)
        examples.append(ex)
    # print("Accuracy {} \nExamples {}".format(accuracies ,examples))

    plt.figure(figsize=(5,5))
    plt.plot(epsilons ,accuracies ,"*-")
    plt.xticks(np.arange(0 ,1.1 ,step=0.1))
    plt.yticks(np.arange(0 ,.35 ,step=0.05))
    plt.title("Accuracy vs Epsilon")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.show()

    # each epsilon plot some example
    cnt = 0
    plt.figure(figsize=(8 ,10))
    for i in range(len(epsilons)):
        for j in range(len(examples[i])):
            cnt +=1
            plt.subplot(len(epsilons) ,len(examples[0]) ,cnt)
            plt.xticks([] ,[])
            plt.yticks([], [])
            if j == 0 :
                plt.ylabel("Eps:{}".format(epsilons[i]) ,fontsize=14)
            orig ,adv ,ex = examples[i][j]
            plt.title("{}->{}".format(orig ,adv))
            plt.imshow(ex ,cmap="gray")
    plt.tight_layout()
    plt.show()


def fgsm_attach(image ,epsilo ,data_grad):
    sign_data_grad = data_grad.sign()
    # distrubance value
    perturbed_image = image + epsilo*sign_data_grad
    # cut range [0 ,1]
    perturbed_image = torch.clamp(perturbed_image ,0 ,1)
    return perturbed_image

def test(model ,device ,test_loader ,epsilon):
    correct = 0
    adv_examples = []
    for data ,target in test_loader:
        data , target = data.to(device),target.to(device)
        # set required_grad properties
        data.requires_grad = True

        # model forward data
        output = model(data)
        init_pred = output.max(1 ,keepdim=True)[1] # get the index of max log-probability

        # initalize error not break distribution
        if init_pred.item() != target.item():
            continue

        # calculate loss
        loss = F.nll_loss(output ,target)
        print('losss ',loss)

        model.zero_grad()
        loss.backward()

        # collect datagrad
        data_grad = data.grad.data


        # FGSM to distribution
        perturbed_data = fgsm_attach(data ,epsilon ,data_grad)
        output = model(perturbed_data)

        # check distribution result
        final_pred = output.max(1 ,keepdim=True)[1] # get the index of the max log-probabolity
        if final_pred.item() == target.item():
            correct += 1
            # debug example epsilon==0
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item() ,final_pred.item() ,adv_ex) )
            # debug some visualization example
            else :
                if len(adv_examples) < 5:
                    adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                    adv_examples.append((init_pred.item() ,final_pred.item() ,adv_ex))

    # calculate epsilon final accuracy
    final_acc = correct / float(len(test_loader))
    print("Epsilon:{}\tTest Accuracy={} / {} = {}.".format(
        epsilon ,correct ,len(test_loader) ,final_acc
    ))
    return final_acc ,adv_examples