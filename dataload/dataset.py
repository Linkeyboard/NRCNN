import os
import torch
from torch.utils.data import Dataset, DataLoader
from .yuv import yuv_import, yuv_10bit_import
import random
import numpy as np
import socket
import cv2


class recyuv(Dataset):
    def __init__(self, rootdir, truthdir, qplist, datatxt):
        self.rootdir = rootdir
        self.truthdir = truthdir
        self.max_qp = 63
        self.datalist = []
        for qp in qplist:
            self.datalist = self.datalist + self.get_data(qp, datatxt)
            
    def __len__(self):
        return len(self.datalist)

    def get_data(self, qp, datatxt):
        f = open(datatxt)
        contents = f.readlines()
        datalist = [(qp, dataname.strip()) for dataname in contents]
        return datalist


    def randomcroptensor(self, x, xtruth, cropsize):
        w = x.shape[2]
        h = x.shape[1]
        rw = random.randint(0, w - cropsize)
        rh = random.randint(0, h - cropsize)
        return x[:, rh : rh + cropsize, rw : rw + cropsize], xtruth[:, rh : rh + cropsize, rw : rw + cropsize]

    def readyuv(self, filename, bit):
        w = 128
        h = 128
        if bit == 8:
            y, u, v = yuv_import(filename, w, h, 1)
        else:
            y, u, v = yuv_10bit_import(filename, w, h, 1)

        y = y.astype(np.float)
        return torch.Tensor(y)

    def __getitem__(self, index):
        index = index % len(self.datalist)
        qp = self.datalist[index][0]
        yuvname = self.datalist[index][1]
        recpath = os.path.join(self.rootdir, str(qp), yuvname)
        orgpath = os.path.join(self.truthdir, yuvname)
        recpatch = self.readyuv(recpath, 8)
        orgpatch = self.readyuv(orgpath, 8)
        qpmap = torch.ones_like(recpatch) * qp / self.max_qp

        return recpatch / 255, orgpatch / 255, qpmap


class rec_chroma(Dataset):
    def __init__(self, rootdir, truthdir, qplist, chroma_idx, datatxt):
        self.rootdir = rootdir
        self.truthdir = truthdir
        self.chroma_idx = chroma_idx
        self.max_qp = 63
        self.datalist = []
        for qp in qplist:
            self.datalist = self.datalist + self.get_data(qp, datatxt)
            
    def __len__(self):
        return len(self.datalist)

    def get_data(self, qp, datatxt):
        f = open(datatxt)
        contents = f.readlines()
        datalist = [(qp, dataname.strip()) for dataname in contents]
        return datalist


    def randomcroptensor(self, x, xtruth, cropsize):
        w = x.shape[2]
        h = x.shape[1]
        rw = random.randint(0, w - cropsize)
        rh = random.randint(0, h - cropsize)
        return x[:, rh : rh + cropsize, rw : rw + cropsize], xtruth[:, rh : rh + cropsize, rw : rw + cropsize]

    def readyuv(self, filename, bit, is_gd):
        w = 128
        h = 128
        if bit == 8:
            y, u, v = yuv_import(filename, w, h, 1)
        else:
            y, u, v = yuv_10bit_import(filename, w, h, 1)
        y, u, v = y.astype(np.float), u.astype(np.float), v.astype(np.float)

        chroma_block = u
        if self.chroma_idx == 1:
            pass
        elif self.chroma_idx == 2:
            chroma_block = v
        
        if is_gd == 1:
            return torch.Tensor(chroma_block)

        chroma_pad_block = cv2.resize(chroma_block.squeeze(), (0, 0), fx = 2, fy = 2, interpolation = cv2.INTER_NEAREST)
        return torch.Tensor(y), torch.Tensor(chroma_block), torch.Tensor(chroma_pad_block).unsqueeze(0)


    def __getitem__(self, index):
        index = index % len(self.datalist)
        qp = self.datalist[index][0]
        yuvname = self.datalist[index][1]
        recpath = os.path.join(self.rootdir, str(qp), yuvname)
        orgpath = os.path.join(self.truthdir, yuvname)
        luma, chroma_rec, chroma_pad = self.readyuv(recpath, 8, 0)
        chroma_gd = self.readyuv(orgpath, 8, 1)
        qpmap = torch.ones_like(chroma_pad) * qp / self.max_qp

        return luma / 255, chroma_rec / 255, chroma_pad / 255, chroma_gd / 255, qpmap

if __name__ == '__main__':
    traindataset = recyuv(r'/gpfs/share/home/1801111388/CNN_LF/data/DIV2K_rec_patch', r'/gpfs/share/home/1801111388/CNN_LF/data/DIV2K_gd_patch', [27], './trainlist.txt')
    trainloader = DataLoader(traindataset, batch_size = 64, shuffle = True, num_workers = 2)
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    print(len(traindataset))
    print(images.shape)
    print(labels.shape)
