
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
from torchvision import datasets, transforms
from tensorboardX import SummaryWriter
from model import Net, ConvNet
from lbfgsb import *
device = torch.device("cpu")
import matplotlib.pyplot as plt


# In[ ]:


np.random.seed(42)
torch.manual_seed(42)


# In[ ]:


# loading the dataset
# note that this time we do not perfrom the normalization operation, see next cell
test_dataset = datasets.MNIST('mnist_data/', train=False, download=True, transform=transforms.Compose(
    [transforms.ToTensor()]
))


# In[ ]:


class Normalize(nn.Module):
    def forward(self, x):
        return (x - 0.1307)/0.3081

# we load the body of the neural net trained last time...
NN1_logits = torch.load('model_conv.net', map_location='cpu') 
NN2_logits = torch.load('model_ff.net', map_location='cpu') 

# ... and add the data normalization as a first "layer" to the network
# this allows us to search for adverserial examples to the real image, rather than
# to the normalized image
NN1_logits = nn.Sequential(Normalize(), NN1_logits)
NN2_logits = nn.Sequential(Normalize(), NN2_logits)

# and here we also create a version of the model that outputs the class probabilities
NN1 = nn.Sequential(NN1_logits, nn.Softmax())
NN2 = nn.Sequential(NN2_logits, nn.Softmax())

# we put the neural net into evaluation mode (this disables features like dropout)
NN1_logits.eval()
NN2_logits.eval()
NN1.eval()
NN2.eval()


# In[ ]:


# define a show function for later
def show(result_gd, result_gd_bounds, result_lbfgsb):    
    gd_s, gd_l, gd_i, gd_t, gd_n = result_gd
    gdb_s, gdb_l, gdb_i, gdb_t, gdb_n = result_gd_bounds
    lbfgsb_s, lbfgsb_l, lbfgsb_i, lbfgsb_t = result_lbfgsb

    def print_res(title, solved, loss, i, time, it=None):
        print(title + ':')
        print('\tSolved:', solved)
        print('\tLoss:', loss)
        print('\tTime:', time, 's')        
        if it is not None:
            print('\tGradient Descent iterations:', it)
        p1 = NN1(torch.from_numpy(i).reshape((1, 1, 28, 28))).detach().numpy()
        p2 = NN2(torch.from_numpy(i).reshape((1, 1, 28, 28))).detach().numpy()
        print('\tNN1_logits class: {} (p = {:.2f}) '.format(p1.argmax(), p1.max()))
        print('\tNN2_logits class: {} (p = {:.2f}) '.format(p2.argmax(), p2.max()))            

    print_res('Gradient Descent', gd_s, gd_l, gd_i, gd_t, gd_n)
    print_res('Gradient Descent w. Bounds', gdb_s, gdb_l, gdb_i, gdb_t, gdb_n)
    print_res('L-BFGS-B', lbfgsb_s, lbfgsb_l, lbfgsb_i, lbfgsb_t)    
    
    f, axarr = plt.subplots(1,3, figsize=(18, 16))
    axarr[0].imshow(gd_i.reshape(28, 28), cmap='gray')
    axarr[0].set_title('Gradient Descent')
    axarr[1].imshow(gdb_i.reshape(28, 28), cmap='gray')
    axarr[1].set_title('Gradient Descent w. Bounds')
    axarr[2].imshow(lbfgsb_i.reshape(28, 28), cmap='gray')
    axarr[2].set_title('L-BFGS-B')    


# In[ ]:


nine = test_dataset[12][0]
fig = plt.figure()
plt.imshow(nine.numpy().reshape((28,28)), cmap='gray')
plt.title('Original Nine')
None # avoid printing title object


# ## Hints:
# 
# - You are given a code skeleton with a few functions, but you will probably need to add some more arugments to the functions and their calls.
# - Split the target "variable" i into a fixed part (where it is equal to nine) and a variable part (that is optimized; set requires_grad for this one). Both parts should be tensors and can be combined into another tensor representing i.
# - Create these two tensors once and then have a function that combines them and calculates the loss.
# - For the loss it is easiest to implement a function implements the loss translation for the less-or-equal ($\leq$) and less ($<$) operators from the lecutre. You can express all parts of the loss function with this. Make this parametric in the choice of d.
# - If implemented correctly your code should not run more than a few seconds.
# - There is no L-BFGS-B optimizer for pytroch yet. We provide a function that uses scipy to do this instead (see below).
# 

# In[ ]:


def solve_gd(max_iter=100, **kwargs):
    t0 = time.time()
    t1 = time.time()
    loss = 0
    solved = False
    
    #Hint:
    # Use pytroch SGD optimizer.
    # Even though it says Stochastic Gradient Descent, we will perfrom Gradient Descent.
    
    #return:
    #solved: Bool; did you find a solution
    #loss: Float; value of loss at the end    
    #i: numpy array; the resulting i
    #t: float; how long the execution took
    #nr: number of iterations
    return solved, loss, nine.detach().numpy(), t1 - t0, 0


# In[ ]:


#feel free to add args to this function
def solve_lbfgsb(**kwargs):
    t0 = time.time()
    t1 = time.time()
    loss = 0
    solved = False

    #Hint:
    # Use the provided lbfgsb(var, min_val, max_val, loss_fn, zero_grad_fn)
    # function.
    # It takes the tensor to optimize (var), the min and max value for each entry (a scalar),
    # a function that returns the current loss-tensor and a function that sets the 
    # gradients of everything used (NN1_logits, NN2_logits) and i_var to zero.
    # This funciton does not return anything but changes var.

    
    #return:
    #solved: Bool; did you find a solution
    #loss: Float; value of loss at the end    
    #i: numpy array; the resulting i
    #t: float; how long the execution took
    return solved, loss, nine.detach().numpy(), t1 - t0


# ## using logits, initialized with zeros

# In[ ]:


show(solve_gd(init_zero=True),
     solve_gd(add_bounds=True, init_zero=True),
     solve_lbfgsb(init_zero=True))


# Note that the first image has weird colors as not all values are in [0, 1].

# ## using logits, initialized with original image

# In[ ]:


show(solve_gd(),
     solve_gd(add_bounds=True),
     solve_lbfgsb())


# ## using probabilites, initialized with zeros

# In[ ]:


show(solve_gd(use_logits=False, init_zero=True),
     solve_gd(use_logits=False, init_zero=True),
     solve_lbfgsb(use_logits=False, init_zero=True))


# ## using probabilites, initialized with original image

# In[ ]:


show(solve_gd(use_prob=True),
     solve_gd(add_bounds=True, use_prob=True),
     solve_lbfgsb(use_prob=True))


# We see that using probabilities is not a viable approach. The numerical optimization problem becomes basically intractable due to the softmax function.

# ## different box constraint (task 1.7; optional), using logits

# In[ ]:


show(solve_gd(box=2),
     solve_gd(add_bounds=True, box=2),
     solve_lbfgsb(box=2))


# Since the region covered by box 2 is mostly emty it does not matter much wether we use init_zero or not.
