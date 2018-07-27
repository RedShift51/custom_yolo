import numpy as np, os, cv2, json, mxnet as mx, time
from mxnet.gluon.data import Dataset, DataLoader
from mxnet.metric import Accuracy
from mxnet import gluon, autograd as ag, ndarray as nd
import matplotlib.pyplot as plt
import loss, utils, model

""" Training net script """

cells, num_anc = 2, 3
PATH = '/home/alex/xplain'
TrainSet = utils.SimpleLoader(path=PATH, imgs_root='imgs', mark_root='markup',
                              mode='Train', out_shape=(15, 19, 19), num_anc=num_anc)
# (15,19,19)
train_loader = DataLoader(TrainSet, batch_size=1, shuffle=True, last_batch='rollover')

cur_model = model.model()
data = mx.sym.Variable(name='data', shape=(1, 3, 300, 300))
out = cur_model.get_symbol(data=data)

net = mx.gluon.SymbolBlock(outputs=out, inputs=data)
ctx = mx.gpu(0)
net.collect_params().initialize(mx.init.Normal(0.01), mx.gpu(0))
#net.load_params(os.path.join(PATH, 'resnet_v1_101-0000.params'), ctx=mx.gpu(0),
#            allow_missing=True, ignore_extra=True)
mx.nd.waitall()
time.sleep(5)
net.load_params(os.path.join(PATH, 'simple-0015.params'), ctx=mx.gpu(0),
            allow_missing=True, ignore_extra=True)
trainer = gluon.Trainer(net.collect_params(), 'adam', {'wd': 0.001, 'lr_scheduler':  utils.PolyScheduler(0.001, 1.5, 65000)})
criterion = loss.totloss(num_anc=num_anc)

logpath = os.path.join(PATH, 'logdir')
if not os.path.exists(logpath):
    os.makedirs(logpath)

metrics, valid_metrics = [loss.MSE(logdir=logpath, tag='train')], [loss.MSE(logdir=logpath, tag='valid')]

num_epochs = 20
for epoch in range(15, num_epochs):
    t0 = time.time()
    total_loss = 0
    #for m in metrics:
    #    m.reset()
    #for m in valid_metrics:
    #    m.reset()
    times = []
    for i0, (img, label) in enumerate(train_loader):
        # print data
        if (i0+1) % 2000 == 0:
            net.export(path=os.path.join(PATH, 'simple'), epoch=epoch)
        batch_size = 1  # data.shape[0]
        dlist = gluon.utils.split_and_load(img, [mx.gpu(0)], 0)
        llist = gluon.utils.split_and_load(label, [mx.gpu(0)], 0)

        if i0 < 13000:
            with ag.record():

                t = time.time()
                preds = [net(X) for X in dlist]
                losses = []
                #if i0 % 50 == 0:
                #    print(preds)
                #    print(llist)

                for i in range(len(preds)):
                    l = criterion(preds[i], llist[i]) #criterion_mse(preds[0][1][i:i+1], llist[0][i][1:]) + \
                    losses.append(l)

            print('batch', i0, 'loss', losses[0][0][0].asnumpy()[0], 'epoch', epoch)

            for l in losses:
                l.backward()
            trainer.step(batch_size)

            for y in metrics:
                y.update(gt_coords=llist, preds=preds)
                y.write_results(int(len(TrainSet.objs) * 0.9 / batch_size * epoch + i0), 'train')
        else:
            with ag.record():

                t = time.time()
                # print(dlist)
                preds = [net(X) for X in dlist]
                losses = []
                #if i0 % 50 == 0:
                #    print(preds)
                #    print(llist)

                for i in range(len(preds)):
                    l = criterion(preds[i], llist[i])  # criterion_mse(preds[0][1][i:i+1], llist[0][i][1:]) + \
                    # criterion_log(preds[0][0][i:i+1], llist[0][i][:1])
                    losses.append(l)

            print('batch', i0, 'val  loss', losses[0][0][0].asnumpy()[0], 'epoch', epoch)

            for y in metrics:
                y.update(gt_coords=llist, preds=preds)
                y.write_results(int(len(TrainSet.objs) * 0.1 / batch_size * epoch + i0 - 9000), 'valid')
