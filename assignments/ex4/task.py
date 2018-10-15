import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time

from torchvision import datasets, transforms
from tensorboardX import SummaryWriter

use_cuda = False
device = torch.device("cuda" if use_cuda else "cpu")
batch_size = 64

np.random.seed(42)
torch.manual_seed(42)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(28*28, 200)
        self.fc2 = nn.Linear(200,10)

    def forward(self, x):
        x = x.view((-1, 28*28))
        x = F.relu(self.fc(x))
        x = self.fc2(x)
        return x
    
train_dataset = datasets.MNIST('mnist_data/', train=True, download=True, transform=transforms.Compose(
    [transforms.ToTensor()]
))
test_dataset = datasets.MNIST('mnist_data/', train=False, download=True, transform=transforms.Compose(
    [transforms.ToTensor()]
))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class Normalize(nn.Module):
    def forward(self, x):
        return (x - 0.1307)/0.3081

# Add the data normalization as a first "layer" to the network
# this allows us to search for adverserial examples to the real image, rather than
# to the normalized image
model = nn.Sequential(Normalize(), Net())

model = model.to(device)
model.train()


def pgd_untargeted(model, x, labels, k, eps, eps_step, clip_min=None, clip_max=None, **kwargs):
    ###############################################
    # TODO fill me
    ###############################################
    return []
    

learning_rate = 0.0001
num_epochs = 20

opt = optim.Adam(params=model.parameters(), lr=learning_rate)

ce_loss = torch.nn.CrossEntropyLoss()

writer = SummaryWriter()
tot_steps = 0


for epoch in range(1,num_epochs+1):
    t1 = time.time()
    for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
        
        ###############################################
        # TODO fill me
        ###############################################
        
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        tot_steps += 1
        opt.zero_grad()
        out = model(x_batch)
        batch_loss = ce_loss(out, y_batch)
        
        if batch_idx % 100 == 0:
            pred = torch.max(out, dim=1)[1]
            acc = pred.eq(y_batch).sum().item() / float(batch_size)
            
            writer.add_scalar('data/accuracy', acc, tot_steps)
            writer.add_scalar('data/loss', batch_loss.item(), tot_steps)
        
        batch_loss.backward()
        opt.step()
        
    tot_test, tot_acc = 0.0, 0.0
    for batch_idx, (x_batch, y_batch) in enumerate(test_loader):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        out = model(x_batch)
        pred = torch.max(out, dim=1)[1]
        acc = pred.eq(y_batch).sum().item()
        tot_acc += acc
        tot_test += x_batch.size()[0]
    t2 = time.time()
        
    print('Epoch %d: Accuracy %.5lf [%.2lf seconds]' % (epoch, tot_acc/tot_test, t2-t1))       
    

###############################################
# TODO fill me
###############################################
