import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchtext import data
from torchtext import datasets
from torchtext.vocab import Vectors, GloVe, CharNGram, FastText
from RNN import Model,Model_2



###############################
# args

parser = argparse.ArgumentParser(description='PyTorch for Sentence Sentiment Classification')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 100)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=0.005, metavar='LR',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--weight_decay', type=float, default=0.00001, metavar='WD',
                    help='weight decay (default: 0.00001)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
criterion = nn.CrossEntropyLoss()  
criterion.cuda()
args = parser.parse_args()
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")

#################################
#functions
def Train(args, model, device, train_loader, optimizer, epoch,criterion):
    model.train()
    cnt = 0;
    for batch in train_loader:
        text, label = batch.text.to(device), (batch.label-1).to(device)  #label: 1-5
        optimizer.zero_grad()
        output = model(text)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
    print('Train Epoch: {}\tLoss: {:.6f}'.format(epoch, loss.item()))


def Test(args, model, device, test_loader,criterion):
    model.eval()
    test_loss = 0
    correct = 0
    n=0
    with torch.no_grad():
        for batch in test_loader:
            n = n+1
            text, label = batch.text.to(device), (batch.label-1).to(device)  #label: 1-5
            output = model(text)
            loss = criterion(output, label)
            test_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(label.view_as(pred)).sum().item()
    test_loss /= n
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, n*args.batch_size,
        100. * correct / (n*args.batch_size)))
    
    return (test_loss, 1.0*correct / (n*args.batch_size))

################################
# DataLoader
################################

# set up fields
TEXT = data.Field()
LABEL = data.Field(sequential=False,dtype=torch.long)

# make splits for data
# DO NOT MODIFY: fine_grained=True, train_subtrees=False
train, val, test = datasets.SST.splits(
    TEXT, LABEL, fine_grained=True, train_subtrees=False)

# print information about the data
print('train.fields', train.fields)
print('len(train)', len(train))
print('vars(train[0])', vars(train[0]))

# build the vocabulary
# you can use other pretrained vectors, refer to https://github.com/pytorch/text/blob/master/torchtext/vocab.py
TEXT.build_vocab(train, vectors=Vectors(name='vector.txt', cache='./data'))
LABEL.build_vocab(train)
# We can also see the vocabulary directly using either the stoi (string to int) or itos (int to string) method.
print(TEXT.vocab.itos[:10])
print(LABEL.vocab.stoi)
print(TEXT.vocab.freqs.most_common(20))

# print vocab information
print('len(TEXT.vocab)', len(TEXT.vocab))
print('TEXT.vocab.vectors.size()', TEXT.vocab.vectors.size())

# make iterator for splits
train_iter, val_iter, test_iter = data.BucketIterator.splits(
    (train, val, test), batch_size=args.batch_size)

# print batch information
batch = next(iter(train_iter)) # for batch in train_iter
print(batch.text) # input sequence
print(batch.label) # groud truth

# Attention: batch.label in the range [1,5] not [0,4] !!!


################################
# After build your network 
################################

# Copy the pre-trained word embeddings we loaded earlier into the embedding layer of our model.
pretrained_embeddings = TEXT.vocab.vectors

print(pretrained_embeddings.shape)

# you should maintain a nn.embedding layer in your network
shape = pretrained_embeddings.shape
#Model():single-direction LSTM, Model_2():bi-direction LSTM
#model = Model(shape[1],shape[0],256) #embeddingDim, embeddingNum, unitNums
model = Model_2(shape[1],shape[0],256)
model.cuda()
model.embedding.weight.data.copy_(pretrained_embeddings)

###################################
# start training and testing
###################################
Loss = [];
Acc = [];

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
print("Start Training:")
for epoch in range(args.epochs):
    Train(args, model, device, train_iter, optimizer, epoch,criterion)
    loss, acc = Test(args, model, device, test_iter,criterion)
    Loss.append(loss)
    Acc.append(acc)

def cut(a,n):
    return [sum(a[i:i+n])/n for i in range(0,len(a),n)]

Loss = cut(Loss,10)
Acc = cut(Acc,10)

print("FINISHED")
print("Loss:")
print(Loss)
print("Accuracy") 
print(Acc)
    


