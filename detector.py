import numpy as np, os, cv2, json, mxnet as mx, time
from mxnet.gluon.data import Dataset, DataLoader
from mxnet.metric import Accuracy
from mxnet import gluon, autograd as ag, ndarray as nd
import matplotlib.pyplot as plt
import loss, utils, model, argparse

cells, num_anc = 2, 3
#PATH = '/home/alex/xplain'
#TrainSet = utils.SimpleLoader(path=PATH, imgs_root='imgs', mark_root='markup',
#                              mode='Train', out_shape=(15, 19, 19), num_anc=num_anc)

#train_loader = DataLoader(TrainSet, batch_size=1, shuffle=True, last_batch='rollover')

def show_boxes(curimg, ancs, thresh, name):
    regs = zip(*np.where(ancs[:3,:,:] > thresh))
    cellcostx, cellcosty = float(curimg.shape[0]) / float(ancs.shape[0]), float(curimg.shape[1]) / float(ancs.shape[1])
    curimg = curimg.astype(np.float32)

    rects = []
    for i0, pair in enumerate(regs):
        #non opencv format
        xstep, ystep = float(curimg.shape[0]) / float(ancs.shape[1]), float(curimg.shape[1]) / float(ancs.shape[2])
        xcenter = xstep * pair[1] + xstep * ancs[3 + pair[0] * 4, pair[1], pair[2]]
        ycenter = ystep * pair[2] + ystep * ancs[3 + pair[0] * 4 + 1, pair[1], pair[2]]
        dx = xstep * ancs[3 + pair[0] * 4 + 2, pair[1], pair[2]]
        dy = ystep * ancs[3 + pair[0] * 4 + 3, pair[1], pair[2]]
        A = (int(xcenter - dx / 2), int(ycenter - dy / 2))
        B = (int(xcenter + dx / 2), int(ycenter + dy / 2))
        rects.append(np.array([[A[0], A[1], B[0], B[1], ancs[pair]]]))
    #curimg = (cv2.cvtColor(curimg, cv2.COLOR_BGR2RGB)*255.).astype(int)
    if len(rects) > 0:
        rects = np.concatenate(rects, 0)
        rects = rects[utils.nms(rects, 0.18)]
        rects = rects.astype(int)
        for i0 in range(len(rects[:,0])):
            cv2.rectangle(curimg, (rects[i0,1], rects[i0,0]), (rects[i0,3], rects[i0,2]), (255,0,0), 4)
    #cv2.imshow('img', cv2.cvtColor(curimg, cv2.COLOR_BGR2RGB))
    curimg = cv2.cvtColor(curimg, cv2.COLOR_BGR2RGB)
    #print(np.max(curimg))
    #curimg = (cv2.cvtColor(curimg, cv2.COLOR_BGR2RGB)).astype(int)
    #cv2.imwrite(name, curimg)
    #print(np.max(curimg))
    cv2.imshow('img', curimg)
    cv2.waitKey()
    return 1

#l = np.array(os.listdir(os.path.join(PATH, 'testing-images')))#[:1000]
#np.random.shuffle(l)
#l = list(l)
#l = ['Pepsi-slogans-over-years.jpg']
cur_model = model.model()
data = mx.sym.Variable(name='data')#shape=(1, 3, img.shape[0], img.shape[1]))
out = cur_model.get_symbol(data=data)

net = mx.gluon.SymbolBlock(outputs=out, inputs=data)
ctx = mx.gpu(0)
net.collect_params().initialize(mx.init.Normal(0.01), mx.gpu(0))
net.load_params(os.path.join('simple-0019.params'), ctx=mx.gpu(0),
                allow_missing=True, ignore_extra=True)
ans_hist = []
#for k0,k in enumerate(l):
parser = argparse.ArgumentParser(description='Detector')
parser.add_argument('path', help='Path to your RGB image')

args = parser.parse_args()

img = cv2.cvtColor(cv2.imread(os.path.join(args.path)), cv2.COLOR_BGR2RGB).astype(float) / 255.
#cv2.imshow('img', img)
#cv2.waitKey()

ans = net(nd.array(np.expand_dims(np.transpose(img, [2,0,1]), 0), ctx=mx.gpu(0)))
anchors = ans.asnumpy()[0,:3,:,:]
coords = ans.asnumpy()[0,3:,:,:]

show_boxes(img, np.concatenate([anchors, coords], 0), thresh=8., name=args.path)
