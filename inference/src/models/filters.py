import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision



def Egde(size=3, channel=1, scale=1e-3):
    if size == 3:
        param = torch.ones((channel, 1, 3, 3), dtype=torch.float16) * (-1)
        for i in range(channel):
            param[i][0][1][1] = 8
        param = nn.Parameter(data=param * scale, requires_grad=False)

    elif size == 5:
        param = torch.ones((channel, 1, 5, 5), dtype=torch.float16) * (-1)
        for i in range(channel):
            param[i][0][1][2] = 2
            param[i][0][2][1] = 2
            param[i][0][2][2] = 4
            param[i][0][2][3] = 2
            param[i][0][3][2] = 2
        param = nn.Parameter(data=param * scale, requires_grad=False)

    else:
        raise NotImplementedError

    return param


def Sobel(size=3, channel=1, scale=1e-3, direction="x"):
    if size == 3:
        param = torch.zeros((channel, 1, 3, 3), dtype=torch.float16)
        if direction == "x":
            for i in range(channel):
                param[i][0][0][0] = param[i][0][2][0] = 1
                param[i][0][0][2] = param[i][0][2][2] = -1
                param[i][0][1][0] = 2
                param[i][0][1][2] = -2
        elif direction == "y":
            for i in range(channel):
                param[i][0][0][0] = param[i][0][0][2] = 1
                param[i][0][2][0] = param[i][0][2][2] = -1
                param[i][0][0][1] = 2
                param[i][0][2][1] = -2
        param = nn.Parameter(data=param * scale, requires_grad=False)

    elif size == 5:
        param = torch.zeros((channel, 1, 5, 5), dtype=torch.float16)
        for i in range(channel):
            param[i][0][0][0] = param[i][0][4][0] = 1
            param[i][0][0][1] = param[i][0][4][1] = 2
            param[i][0][0][3] = param[i][0][4][3] = -2
            param[i][0][0][4] = param[i][0][4][4] = -1

            param[i][0][1][0] = param[i][0][3][0] = 4
            param[i][0][1][1] = param[i][0][3][1] = 8
            param[i][0][1][3] = param[i][0][3][3] = -8
            param[i][0][1][4] = param[i][0][3][4] = -4

            param[i][0][2][0] = 6
            param[i][0][2][1] = 12
            param[i][0][2][3] = -12
            param[i][0][2][4] = -6
        param = nn.Parameter(data=param * scale, requires_grad=False)

    else:
        raise NotImplementedError

    return param

    # if direction == 'x':
    #     return param
    # else:
    #     # param = param.transpose(3, 2)
    #     return param


def Sobel_xy(size=3, channel=1, scale=1e-3, direction="xy"):
    param = torch.zeros((channel, 1, 3, 3), dtype=torch.float16)
    if size == 3 and direction == "xy":
        for i in range(channel):
            param[i][0][0][1] = 1
            param[i][0][0][2] = 2
            param[i][0][1][0] = -1
            param[i][0][1][2] = 1
            param[i][0][2][0] = -2
            param[i][0][2][1] = -1
        param = nn.Parameter(data=param * scale, requires_grad=False)

    elif size == 3 and direction == "yx":
        for i in range(channel):
            param[i][0][0][0] = -2
            param[i][0][0][1] = -1
            param[i][0][1][0] = -1
            param[i][0][1][2] = 1
            param[i][0][2][1] = 1
            param[i][0][2][2] = 2
        param = nn.Parameter(data=param * scale, requires_grad=False)

    else:
        raise NotImplementedError

    return param


def Roberts(size=3, channel=1, scale=1e-3, direction="x"):
    if size == 3 and direction == "x":
        param = torch.zeros((channel, 1, 3, 3), dtype=torch.float16)
        for i in range(channel):
            param[i][0][0][0] = 1
            param[i][0][1][1] = -1
        param = nn.Parameter(data=param * scale, requires_grad=False)

    elif size == 3 and direction == "y":
        param = torch.zeros((channel, 1, 3, 3), dtype=torch.float16)
        for i in range(channel):
            param[i][0][0][1] = 1
            param[i][0][1][0] = -1
        param = nn.Parameter(data=param * scale, requires_grad=False)

    elif size == 2 and direction == "x":
        param = torch.zeros((channel, 1, 2, 2), dtype=torch.float16)
        for i in range(channel):
            param[i][0][0][0] = 1
            param[i][0][1][1] = -1
        param = nn.Parameter(data=param * scale, requires_grad=False)

    elif size == 2 and direction == "y":
        param = torch.zeros((channel, 1, 2, 2), dtype=torch.float16)
        for i in range(channel):
            param[i][0][0][1] = 1
            param[i][0][1][0] = -1
        param = nn.Parameter(data=param * scale, requires_grad=False)

    else:
        raise NotImplementedError

    return param


def Prewitt(size=3, channel=1, scale=1e-3, direction="x"):
    param = torch.zeros((channel, 1, 3, 3), dtype=torch.float16)
    if size == 3 and direction == "y":
        for i in range(channel):
            param[i][0][0][0] = -1
            param[i][0][1][0] = -1
            param[i][0][2][0] = -1
            param[i][0][0][2] = 1
            param[i][0][1][2] = 1
            param[i][0][2][2] = 1
        param = nn.Parameter(data=param * scale, requires_grad=False)

    elif size == 3 and direction == "x":
        for i in range(channel):
            param[i][0][0][0] = -1
            param[i][0][0][1] = -1
            param[i][0][0][2] = -1
            param[i][0][2][0] = 1
            param[i][0][2][1] = 1
            param[i][0][2][2] = 1
        param = nn.Parameter(data=param * scale, requires_grad=False)

    elif size == 3 and direction == "xy":
        for i in range(channel):
            param[i][0][0][1] = 1
            param[i][0][0][2] = 1
            param[i][0][1][0] = -1
            param[i][0][1][2] = 1
            param[i][0][2][0] = -1
            param[i][0][2][1] = -1
        param = nn.Parameter(data=param * scale, requires_grad=False)

    elif size == 3 and direction == "yx":
        for i in range(channel):
            param[i][0][0][0] = -1
            param[i][0][0][1] = -1
            param[i][0][1][0] = -1
            param[i][0][1][2] = 1
            param[i][0][2][1] = 1
            param[i][0][2][2] = 1
        param = nn.Parameter(data=param * scale, requires_grad=False)

    else:
        raise NotImplementedError

    return param


def Laplacian(channel=1, scale=1e-3, type=1):
    param = torch.ones((channel, 1, 3, 3), dtype=torch.float16)
    if type == 1:
        for i in range(channel):
            param[i][0][0][0] = 0
            param[i][0][0][2] = 0
            param[i][0][1][1] = -4
            param[i][0][2][0] = 0
            param[i][0][2][2] = 0
        param = nn.Parameter(data=param * scale, requires_grad=False)
    else:
        for i in range(channel):
            param[i][0][1][1] = -4
        param = nn.Parameter(data=param * scale, requires_grad=False)
    return param


def HighPass(x, kernel_size=15, sigma=5):
    filter2 = torchvision.transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
    gauss = filter2(x)
    return x - gauss


class AdaptiveFilters(nn.Module):
    def __init__(self, dim, training=True):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.training = training
        self.dim = dim
        self.Sobel_x = Sobel(channel=dim, direction="x")
        self.Laplation = Laplacian(channel=dim)
        self.Edge = Egde(channel=dim)
        self.Roberts_x = Roberts(channel=dim, direction="x")
        self.Roberts_y = Roberts(channel=dim, direction="y")
        self.Sobel_xy = Sobel_xy(channel=dim, direction="xy")
        self.Sobel_yx = Sobel_xy(channel=dim, direction="yx")
        self.alpha = nn.Parameter(
            torch.ones_like(torch.FloatTensor(9)).requires_grad_()
        )
        self.beta = nn.Parameter(
            torch.zeros_like(torch.FloatTensor(1)).requires_grad_()
        )
        self.Sobel_y = Sobel(channel=dim, direction="y")
        self.weight = None
        self.isCombine = False

    def Combine(self):
        if not self.training:
            self.weight = (
                self.Sobel_x * self.alpha[0]
                + self.Sobel_y * self.alpha[1]
                + self.Laplation * self.alpha[2]
                + self.Edge * self.alpha[3]
                + self.Roberts_x * self.alpha[4]
                + self.Roberts_y * self.alpha[5]
                + self.Sobel_xy * self.alpha[6]
                + self.Sobel_yx * self.alpha[7]
            )
            self.weight = self.weight.to(torch.float16)
            self.__delattr__("Sobel_x")
            self.__delattr__("Sobel_y")
            self.__delattr__("Laplation")
            self.__delattr__("Edge")
            self.__delattr__("Roberts_x")
            self.__delattr__("Roberts_y")
            self.__delattr__("Sobel_xy")
            self.__delattr__("Sobel_yx")

        self.isCombine = True

    def forward(self, x):
        # if not self.training and not self.isCombine:
        #     self.Combine()

        if self.weight is None:
            Sobel_x = (
                F.conv2d(
                    input=x, weight=self.Sobel_x, stride=1, groups=self.dim, padding=1
                )
                * self.alpha[0]
            )
            Sobel_y = (
                F.conv2d(
                    input=x, weight=self.Sobel_y, stride=1, groups=self.dim, padding=1
                )
                * self.alpha[1]
            )
            Laplation = (
                F.conv2d(
                    input=x, weight=self.Laplation, stride=1, groups=self.dim, padding=1
                )
                * self.alpha[2]
            )
            Egde = (
                F.conv2d(
                    input=x, weight=self.Edge, stride=1, groups=self.dim, padding=1
                )
                * self.alpha[3]
            )
            Sobel_xy = (
                F.conv2d(
                    input=x, weight=self.Sobel_xy, stride=1, groups=self.dim, padding=1
                )
                * self.alpha[4]
            )
            Sobel_yx = (
                F.conv2d(
                    input=x, weight=self.Sobel_yx, stride=1, groups=self.dim, padding=1
                )
                * self.alpha[5]
            )
            Roberts_x = (
                F.conv2d(
                    input=x, weight=self.Roberts_x, stride=1, groups=self.dim, padding=1
                )
                * self.alpha[6]
            )
            Roberts_y = (
                F.conv2d(
                    input=x, weight=self.Roberts_y, stride=1, groups=self.dim, padding=1
                )
                * self.alpha[7]
            )
            high_pass = HighPass(x) * self.alpha[8]
            return (
                Sobel_x
                + Sobel_y
                + Laplation
                + Egde
                + x * self.beta[0]
                + Sobel_xy
                + Sobel_yx
                + Roberts_x
                + Roberts_y
                + high_pass
            )
        else:
            out = F.conv2d(
                input=x,
                weight=self.weight.cuda().to(torch.float16),
                stride=1,
                groups=self.dim,
                padding=1,
            )
            return HighPass(x) * self.alpha[8] + out + x * self.beta[0]
