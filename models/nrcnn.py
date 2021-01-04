import torch
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable

class ResBlock(torch.nn.Module):
    def __init__(self, kernalnum = 64, conv = 3, pd = 1):
        super(ResBlock, self).__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(kernalnum, kernalnum, conv, 1, pd),
            torch.nn.ReLU(inplace = True),
            torch.nn.Conv2d(kernalnum, kernalnum, conv, 1, pd),
        )
    def forward(self, x):
        out = self.block(x)
        out = out + x
        return out

class NRCNN(torch.nn.Module):
    def __init__(self, blocknum, kernalnum = 64, conv = 3):
        if conv == 3:
            pd = 1
        elif conv == 5:
            pd = 2
        elif conv == 7:
            pd = 3
        super(NRCNN, self).__init__()
        self.blocknum = blocknum
        self.part1 = torch.nn.Sequential(
            torch.nn.Conv2d(2, kernalnum, conv, 1, pd),
            torch.nn.ReLU(inplace = True),
        )
        self.part2 = torch.nn.ModuleList()
        for i in range(blocknum):
            self.part2.append(ResBlock(kernalnum, conv, pd))
        self.part3 = torch.nn.Sequential(
            torch.nn.Conv2d(kernalnum, kernalnum, conv, 1, pd),
            torch.nn.ReLU(inplace = True),
            torch.nn.Conv2d(kernalnum, 1, conv, 1, pd),
        )

    def forward(self, x):
        out = self.part1(x)
        for layer in self.part2:
            out = layer(out)
        out = self.part3(out)
        out = out + x[:, 0, :, :].unsqueeze(1)
        return out
        
class Extened_NRCNN(torch.nn.Module):
    def __init__(self, blocknum, kernalnum):
        super(Extened_NRCNN, self).__init__()
        self.part1_1 = torch.nn.Sequential(
            torch.nn.Conv2d(2, kernalnum // 2, 3, 1, 1),
            torch.nn.ReLU(inplace = True),
        )
        self.part1_2 = torch.nn.Sequential(
            torch.nn.Conv2d(1, kernalnum // 2, 3, 1, 1),
            torch.nn.ReLU(inplace = True),
        )
        self.part2 = torch.nn.ModuleList()
        for i in range(blocknum):
            self.part2.append(ResBlock(kernalnum))
        self.part3 = torch.nn.Sequential(
            torch.nn.Conv2d(kernalnum, kernalnum, 3, 1, 1),
            torch.nn.ReLU(inplace = True),
            torch.nn.Conv2d(kernalnum, 1, 3, 1, 1),
            torch.nn.AvgPool2d(2, 2),
        )
    def forward(self, u, y):
        out1 = self.part1_1(u)
        out2 = self.part1_2(y)
        out = torch.cat((out1, out2), 1)
        for layer in self.part2:
            out = layer(out)
        out = self.part3(out)
        return out


if __name__ == '__main__':
    NRCNN = NRCNN(8, 64)
    NRCNN.cuda()
    a = torch.ones(16, 1, 64, 64)
    a = Variable(a.cuda())
    output = NRCNN(a)
    print(output.size(), output)
