import numpy as np
import torch
from os.path import join, split, isdir, isfile, split, abspath, dirname
import os
import torch.nn as nn
from dataload.dataset import recyuv
from torch.optim import lr_scheduler
from models.nrcnn import NRCNN
import torch.nn.functional as F
import argparse
import sys, time
from utils import Logger, Averagvalue, save_checkpoint, psnr
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import socket

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--batch_size', default=64, type=int, metavar='BT', help='batch size')
parser.add_argument('--res_block', default=4, type=int, metavar='RB', help='the number of residual block')
# =============== optimizer
parser.add_argument('--lr', '--learning_rate', default=1e-4, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--weight_decay', '--wd', default=0, type=float, metavar='W', help='default weight decay')
parser.add_argument('--stepsize', default=400, type=int, metavar='SS', help='learning rate step size')
parser.add_argument('--gamma', '--gm', default=0.1, type=float, help='learning rate decay parameter: Gamma')
parser.add_argument('--maxepoch', default=400, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--itersize', default=1, type=int, metavar='IS', help='iter size')
# =============== misc
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--print_freq', '-p', default=100, type=int, metavar='N', help='print frequency (default: 100)')
parser.add_argument('--gpu', default='0', type=str, help='GPU ID')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--save_path', help='path to save checkpoint', default='./checkpoint')
parser.add_argument('--show_path', help='path to save data for tensorboard', type=str, default='./plot')
# ================ qp
parser.add_argument('--qp_start', help='start qp', default=45, type=int)
parser.add_argument('--qp_end', help='end qp', default=45, type=int)


args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

qplist = [qp for qp in range(args.qp_start, args.qp_end + 1)]
traindataset = recyuv(r'L:\Dataset\DIV2K\hpm\patch', r'L:\Dataset\DIV2K\org\patch', qplist, './trainlist.txt')
testdataset = recyuv(r'L:\Dataset\DIV2K\hpm\patch', r'L:\Dataset\DIV2K\org\patch', qplist, './testlist.txt')
testloader = DataLoader(testdataset, batch_size = 32, shuffle = False, num_workers = 0)
trainloader = DataLoader(traindataset, batch_size = args.batch_size, shuffle = True, num_workers = 0)

def main():
    # model
    model = NRCNN(args.res_block, 64)
    model.cuda()
    #model.apply(weights_init)
    if args.resume:
        if isfile(args.resume): 
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}'"
                  .format(args.resume))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    
    #tune lr
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma)
    
    # log
    if not isdir(args.save_path):
        os.makedirs(args.save_path)
    log = Logger(join(args.save_path, '%s-%d-log.txt' %('sgd',args.lr)))
    sys.stdout = log

    for epoch in range(args.start_epoch, args.maxepoch):
        if epoch == 0:
            print("Performing initial testing...")

        train(trainloader, model, optimizer, epoch, 
            save_dir = join(args.save_path, 'epoch-%d-training-record' % epoch))
        log.flush() # write log
        scheduler.step() # will adjust learning rate

    writer.close()
        
        
def train(trainloader, model, optimizer, epoch, save_dir):
    global_step = epoch * len(trainloader) // args.print_freq
    batch_time = Averagvalue()
    data_time = Averagvalue()
    loss_list = Averagvalue()
    model.train()
    for i, (image, label, qpmap) in enumerate(trainloader):
        image, label, qpmap = image.cuda(), label.cuda(), qpmap.cuda()
        outputs = model(torch.cat([image, qpmap], 1))

        psnr_1 = psnr(F.mse_loss(image, label).item())
        psnr_2 = psnr(F.mse_loss(outputs, label).item())

        loss = F.mse_loss(outputs, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_list.update(loss.item(), image.size(0))

        if i % args.print_freq  == args.print_freq - 1:
            info = 'Epoch: [{0}/{1}][{2}/{3}] '.format(epoch, args.maxepoch, i, len(trainloader)) + \
                    'Loss {loss.val:f} (avg:{loss.avg:f})'.format(loss = loss_list) + ' PSNR {:.4f}'.format(psnr_2 - psnr_1)
            print(info)
            
            writer = SummaryWriter(args.show_path)
            writer.add_scalar('scalar/loss', loss_list.avg, global_step)
            delta_psnr = test_luma(model)
            writer.add_scalar('scalar/psnr', delta_psnr, global_step)
            writer.close()

            global_step += 1
            loss_list.reset()

    if not isdir(save_dir):
        os.makedirs(save_dir)
    save_checkpoint({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
        }, filename = join(save_dir, "epoch-%d-checkpoint.pth" % epoch))

def test_luma(model):
    psnr_before = Averagvalue()
    psnr_after = Averagvalue()
    for i, (image, label, qpmap) in enumerate(testloader):
        image, label, qpmap = image.cuda(), label.cuda(), qpmap.cuda()
        outputs = model(torch.cat([image, qpmap], 1))
        psnr_1 = psnr(F.mse_loss(image, label).item())
        psnr_2 = psnr(F.mse_loss(outputs, label).item())

        info = '[{}]'.format(i) + 'PSNR from {:.4f} to {:.4f}'.format(psnr_1, psnr_2) + ' Delta:{:.4f}'.format(psnr_2 - psnr_1)
        psnr_before.update(psnr_1)
        psnr_after.update(psnr_2)
    return psnr_after.avg - psnr_before.avg

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        # xavier(m.weight.data)
        m.weight.data.normal_(0, 0.01)
        if m.weight.data.shape == torch.Size([1, 5, 1, 1]):
            # for new_score_weight
            torch.nn.init.constant_(m.weight, 0.2)
        if m.bias is not None:
            m.bias.data.zero_()

if __name__ == "__main__":
    main()
