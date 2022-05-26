# import torch
# from torchvision import transforms
#
# xc = torch.linspace(0, 1, 100)
# xx, yy = torch.meshgrid(xc, xc)
# print(xx.shape,yy.shape)
# #xx = xx.reshape(-1, 1)
# #yy = yy.reshape(-1, 1)
# print(xx.shape,yy.shape)
# xy = torch.cat([xx, yy], dim=1)
#
# unloader = transforms.ToPILImage()
# image = unloader(xx)
# image.save('example.jpg')
#u_pred = u(xy)

import numpy as np
import matplotlib.pyplot as plt
import torch
import numpy

import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Domain and Sampling
def interior(n=1000):
    x = torch.rand(n, 1)
    y = torch.rand(n, 1)
    cond = (2 - x ** 2) * torch.exp(-y)
    return x.requires_grad_(True), y.requires_grad_(True), cond


def down_yy(n=100):
    x = torch.rand(n, 1)
    y = torch.zeros_like(x)
    cond = x ** 2
    return x.requires_grad_(True), y.requires_grad_(True), cond


def up_yy(n=100):
    x = torch.rand(n, 1)
    y = torch.ones_like(x)
    cond = x ** 2 / torch.exp(torch.tensor([1]))
    return x.requires_grad_(True), y.requires_grad_(True), cond


def down(n=100):
    x = torch.rand(n, 1)
    y = torch.zeros_like(x)
    cond = x ** 2
    return x.requires_grad_(True), y.requires_grad_(True), cond


def up(n=100):
    x = torch.rand(n, 1)
    y = torch.ones_like(x)
    cond = x ** 2 / torch.exp(torch.tensor([1]))
    return x.requires_grad_(True), y.requires_grad_(True), cond


def left(n=100):
    y = torch.rand(n, 1)
    x = torch.zeros_like(y)
    cond = torch.zeros_like(x)
    return x.requires_grad_(True), y.requires_grad_(True), cond


def right(n=100):
    y = torch.rand(n, 1)
    x = torch.ones_like(y)
    cond = torch.exp(-y)
    return x.requires_grad_(True), y.requires_grad_(True), cond


# Neural Network
class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)


# Loss
loss = torch.nn.MSELoss()


def gradients(u, x, order=1):
    if order == 1:
        return torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                   create_graph=True,
                                   only_inputs=True, )[0]
    else:
        return gradients(gradients(u, x), x, order=order - 1)


def l_interior(u):
    x, y, cond = interior()
    uxy = u(torch.cat([x, y], dim=1))
    return loss(gradients(uxy, x, 2) - gradients(uxy, y, 4), cond)


def l_down_yy(u):
    x, y, cond = down_yy()
    uxy = u(torch.cat([x, y], dim=1))
    return loss(gradients(uxy, y, 2), cond)


def l_up_yy(u):
    x, y, cond = up_yy()
    uxy = u(torch.cat([x, y], dim=1))
    return loss(gradients(uxy, y, 2), cond)


def l_down(u):
    x, y, cond = down()
    uxy = u(torch.cat([x, y], dim=1))
    return loss(uxy, cond)


def l_up(u):
    x, y, cond = up()
    uxy = u(torch.cat([x, y], dim=1))
    return loss(uxy, cond)


def l_left(u):
    x, y, cond = left()
    uxy = u(torch.cat([x, y], dim=1))
    return loss(uxy, cond)


def l_right(u):
    x, y, cond = right()
    uxy = u(torch.cat([x, y], dim=1))
    return loss(uxy, cond)


# Training

u = MLP()
#u = u.to(device)
opt = torch.optim.Adam(params=u.parameters())
for i in range(10000):
    opt.zero_grad()
    l = l_interior(u) \
        + l_up_yy(u) \
        + l_down_yy(u) \
        + l_up(u) \
        + l_down(u) \
        + l_left(u) \
        + l_right(u)
    l.backward()
    opt.step()
    print(i)

xc = torch.linspace(0, 1, 100)
xx, yy = torch.meshgrid(xc, xc)
xx = xx.reshape(-1, 1)
yy = yy.reshape(-1, 1)
xy = torch.cat([xx, yy], dim=1)
u_pred = u(xy)
u_pred = u_pred.detach().numpy()
u_pred = u_pred.reshape((100,100)).T

x= torch.linspace(0,1,100)
y= torch.linspace(0,1,100)
X,Y = numpy.meshgrid(x,y)
pic=plt.contourf(X,Y,u_pred, 500,cmap='jet')
plt.colorbar(pic)
plt.title('Pred')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
from mpl_toolkits.mplot3d import Axes3D
# x= torch.linspace(0,1,100)
# y= torch.linspace(0,1,100)
# X,Y = numpy.meshgrid(x,y)
# Z = X*X*numpy.exp(-Y)
# pic=plt.contourf(X,Y,Z, 500,cmap='jet')
# plt.colorbar(pic)
# # plt.legend('1')
# plt.title('True')
# x_stick = numpy.arange(0,1,10)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()





x= torch.linspace(0,1,100)
y= torch.linspace(0,1,100)
X,Y = numpy.meshgrid(x,y)
Z = X*X*numpy.exp(-Y)
pic=plt.contourf(X,Y,Z, 500,cmap='jet')
plt.colorbar(pic)
# plt.legend('1')
plt.title('True')
x_stick = numpy.arange(0,1,10)
plt.xlabel('x')
plt.ylabel('y')
plt.show()
