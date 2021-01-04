import numpy as np
import torch
import os
from os.path import join, split, isdir, isfile
import torch.nn as nn
import torchvision
import argparse
import torch.nn.functional as F
from models.nrcnn import NRCNN
from utils import Logger, Averagvalue, psnr
import sys
import socket
from dataload.dataset import recyuv
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter



def main():
    args.cuda = True
    model = NRCNN(4, 64)
    model.cuda()
    if isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}'".format(args.resume))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

    net = model

    log = Logger('test_result.txt')
    sys.stdout = log
    test(model)
    log.flush()

def write_log():
    writer = SummaryWriter()
    args.cuda = True
    model = NRCNN(4, 64)
    model.cuda()
    for epoch in range(args.start_epoch, args.end_epoch + 1):
        resume = './tmp/RCF/checkpoint_epoch' + str(epoch) + '.pth'
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint['state_dict'])
        deltapsnr = test(model)
        writer.add_scalar('scalar/psnr', deltapsnr, epoch)

    writer.close()


def test_luma(model, testloader):
    psnr_before = Averagvalue()
    psnr_after = Averagvalue()
    for i, (image, label) in enumerate(testloader):
        image, label = image.cuda(), label.cuda()
        outputs = model(image)
        losslist = []
        losslist.append(F.mse_loss(label, image).item())
        loss = F.mse_loss(outputs, label)
        losslist.append(loss.item())

        info = '[{}]'.format(i) + 'PSNR from {:.4f} to {:.4f}'.format(psnr(losslist[0]), psnr(losslist[-1])) + ' Delta:{:.4f}'.format(psnr(losslist[-1])- psnr(losslist[0]))
        psnr_before.update(psnr(losslist[0]))
        psnr_after.update(psnr(losslist[-1]))

    #print('PSNR from {:.4f} to {:.4f}'.format(psnr_before.avg, psnr_after.avg))
    return psnr_after.avg - psnr_before.avg


def test_chroma(model, testloader):
    psnr_before = Averagvalue()
    psnr_after = Averagvalue()
    for i, (luma, chroma_rec, chroma_pad, chroma_gd) in enumerate(testloader):
        luma, chroma_rec, chroma_pad, chroma_gd = luma.cuda(), chroma_rec.cuda(), chroma_pad.cuda(), chroma_gd.cuda()
        outputs = model(luma, chroma_pad)
        losslist = []
        losslist.append(F.mse_loss(chroma_rec, chroma_gd).item())
        loss = F.mse_loss(outputs, chroma_gd - chroma_rec)
        losslist.append(loss.item())

        info = '[{}]'.format(i) + 'PSNR from {:.4f} to {:.4f}'.format(psnr(losslist[0]), psnr(losslist[-1])) + ' Delta:{:.4f}'.format(psnr(losslist[-1])- psnr(losslist[0]))
        psnr_before.update(psnr(losslist[0]))
        psnr_after.update(psnr(losslist[-1]))

    #print('PSNR from {:.4f} to {:.4f}'.format(psnr_before.avg, psnr_after.avg))
    return psnr_after.avg - psnr_before.avg

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Pytorch Testing')
    parser.add_argument('--resume', default = '', type = str, metavar = 'PATH', help = 'path to latest checkpoint(default: none)')
    parser.add_argument('--gpu', default = '0', type = str, metavar = 'N', help = 'GPU ID')
    parser.add_argument('--start_epoch', default = 0, type = int, metavar = 'N', help = 'start epoch')
    parser.add_argument('--end_epoch', default = 0, type = int, metavar = 'N', help = 'end epoch')
    parser.add_argument('--writer', default = 0, type = int, metavar = 'N', help = 'print psnr changes')

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.writer:
        write_log()
    else:
        main()
