from yuv import yuv_import, yuv_export, yuv_10bit_import, yuv_10bit_export
import os
import random

def splitrecyuv(yuvdir, splitdir):
    if not os.path.exists(splitdir):
        os.makedirs(splitdir)
    for yuvname in os.listdir(yuvdir):
        w = int(yuvname.split('_')[-2])
        h = int(yuvname.split('_')[-1][: -4])
        '''
        w = int(yuvname.split('x')[0].split('_')[-1])
        h = int(yuvname.split('x')[1].split('_')[0])
        '''
        print(yuvname, w, h)
        y, u, v = yuv_10bit_import(os.path.join(yuvdir, yuvname), w, h, 1)

        blocksize = 128
        blocksizeuv = int(blocksize / 2)
        idx = 1
        for i in range(int(h / blocksize)):
            starth = i * blocksize
            startuv = i * blocksizeuv
            for j in range(int(w / blocksize)):
                ty = y[:, starth : starth + blocksize, j * blocksize : (j + 1) * blocksize]
                tu = u[:, startuv : startuv + blocksizeuv, j * blocksizeuv : (j + 1) * blocksizeuv]
                tv = v[:, startuv : startuv + blocksizeuv, j * blocksizeuv : (j + 1) * blocksizeuv]
                writename = yuvname.split('.')[-2] + '_' + str(idx) + '.yuv'
                yuv_10bit_export(os.path.join(splitdir, writename), ty, tu, tv)
                idx = idx + 1

def get_random_datalist(datadir):
    datalist = os.listdir(datadir)
    random.shuffle(datalist)
    f = open(r'./testlist.txt', 'w')
    for data in datalist[ : 7430]:
        f.write('{}\n'.format(data))
    f.close()
    f = open(r'./trainlist.txt', 'w')
    for data in datalist[7430 : ]:
        f.write('{}\n'.format(data))
    f.close()

if __name__ == "__main__":
    '''
    qplist = [27, 32, 38, 45]
    splitdir = r'/gpfs/share/home/1801111388/CNN_LF/data/DIV2K_rec_patch'
    yuvdir = r'/gpfs/share/home/1801111388/CNN_LF/data/DIV2K_rec'
    #splitrecyuv(yuvdir, splitdir)
    for qp in qplist:
        splitrecyuv(os.path.join(yuvdir, str(qp)), os.path.join(splitdir, str(qp)))
    '''
    get_random_datalist(r'L:\Dataset\DIV2K\org\patch')
