import torch
import torch.optim as optim
import torch.nn as nn

torch.manual_seed(1)
alpha = 1
#alpha = 2
C = 0
mom=0
#mom=.1
damp=0

##############################
## encode the dataset to fit the one specified in HW4.pdf (note that bias
## is part of the network now)
## Dimensions: X (2x3); y (3)
##############################
X = torch.Tensor([[1,0,0],[0,1,0]])
print("X ", X)
y = torch.LongTensor([0,1,2])
print("Y " , y)

class ShallowNet(nn.Module):
    def __init__(self):
        super(ShallowNet, self).__init__()
        self.fc1 = nn.Linear(2,3, bias=True)
    
    def forward(self, X):
        return self.fc1(X)

net = ShallowNet()
print(net)

print(net(torch.transpose(X,0,1)).squeeze())

optimizer = optim.SGD(net.parameters(), lr=alpha, momentum=mom, dampening=damp, weight_decay=C, nesterov=False)
optimizer.zero_grad()

criterion = nn.CrossEntropyLoss()

for iter in range(10000):
    netOutput = net(torch.transpose(X,0,1))

    ##############################
    ## provide the arguments for the criterion function
    ## Dimensions: loss (scalar)
    ##############################    
    loss = criterion(netOutput,y)

    loss.backward()
    gn = 0
    for f in net.parameters():
        gn = gn + torch.norm(f.grad)
    if iter % 1000 == 0:
        print("Iter: %d, Loss: %f; ||g||: %f" % (iter, loss, gn))

    ##############################
    ## Use two functions within the optimizer instance to perform the update step
    ##############################    
    optimizer.step()
    optimizer.zero_grad()

for f in net.parameters():
    print(f)

print(nn.Softmax(dim=1)(net(torch.transpose(X, 0, 1))))

