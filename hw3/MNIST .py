from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from plot import plot_loss_and_acc

class MLPNet1(nn.Module):
    def __init__(self):
        super(MLPNet1,self).__init__()
        self.fc1 = nn.Linear(784,128)
        self.fc2 = nn.Linear(128,10)
    
    def forward(self, x):
        x = x.view(-1,28*28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class MLPNet2(nn.Module):
    def __init__(self):
        super(MLPNet2,self).__init__()
        self.fc1 = nn.Linear(784,256)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,10)
    
    def forward(self, x):
        x = x.view(-1,28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ConvNet1(nn.Module):  # same shape
    def __init__(self):
        super(ConvNet1, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size = (3,3), stride=(1,1), padding=(1,1))
        self.conv2 = nn.Conv2d(8, 16, kernel_size = (3,3), stride=(1,1), padding=(1,1))
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ConvNet2(nn.Module):  # valid shape
    def __init__(self):
        super(ConvNet2, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 784)
        self.fc2 = nn.Linear(784, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
def train(args, model, device, train_loader, optimizer, epoch,criterion):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader,criterion):
    model.eval()
    test_loss = 0
    correct = 0
    n=0
    with torch.no_grad():
        for data, target in test_loader:
            n = n+1
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= n

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    return (test_loss, 1.0*correct / len(test_loader.dataset))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 100)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--weight_decay', type=float, default=0.00, metavar='WD',
                        help='weight decay (default: 0)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=60, metavar='N',
                        help='how many batches to wait before logging training status')
    
    criterion = nn.CrossEntropyLoss()  
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
    
                                        
    # change Net model here
    model = MLPNet1().to(device)
    MLP1_loss = [];
    MLP1_acc = [];
    
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    print("Start Training: MLP model with one hidden layer")
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch,criterion)
        loss, acc = test(args, model, device, test_loader,criterion)
        MLP1_loss.append(loss)
        MLP1_acc.append(acc)
    
    # change Net model here
    model = MLPNet2().to(device)
    MLP2_loss = [];
    MLP2_acc = [];
    
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    print("Start Training: MLP model with two hidden layer")
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch,criterion)
        loss, acc = test(args, model, device, test_loader,criterion)
        MLP2_loss.append(loss)
        MLP2_acc.append(acc)


    # change Net model here
    model = ConvNet1().to(device)
    Conv1_loss = [];
    Conv1_acc = [];
    
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    print("Start Training: ConvNet model 1")
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch,criterion)
        loss, acc = test(args, model, device, test_loader,criterion)
        Conv1_loss.append(loss)
        Conv1_acc.append(acc)

     # change Net model here
    model = ConvNet2().to(device)
    Conv2_loss = [];
    Conv2_acc = [];
    
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    print("Start Training: ConvNet model 2")
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch,criterion)
        loss, acc = test(args, model, device, test_loader,criterion)
        Conv2_loss.append(loss)
        Conv2_acc.append(acc)

    # copy the result to jupyter notebook and plot
    print("MLP1_loss = ") 
    print(MLP1_loss)
    print("MLP1_acc = ") 
    print(MLP1_acc)
    print("MLP2_loss = ")
    print(MLP2_loss)
    print("MLP2_acc = " )
    print(MLP2_acc)
    print("Conv1_loss = ") 
    print(Conv1_loss)
    print("Conv1_acc = ") 
    print(Conv1_acc)
    print("Conv2_loss = ")
    print(Conv2_loss)
    print("Conv2_acc = " )
    print(Conv2_acc)
    
if __name__ == '__main__':
    main()