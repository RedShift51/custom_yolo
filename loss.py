import mxnet as mx
from mxnet import gluon
from mxnet import nd
import mxnet as mx
import numpy as np
import tensorflow as tf

""" Custom loss function class """

class totloss(gluon.loss.Loss):

    def __init__(self, num_anc, **kwargs):
        super(totloss, self).__init__(None, 0, **kwargs)
        self.class_error_func = gluon.loss.SoftmaxCrossEntropyLoss()
        self.num_anc = num_anc
        self.batch_size = 1
        #self.num_cls = num_cls

    def split_signals(self, pred, label):
        pred_anc = pred[:, :self.num_anc, :, :]
        pred_coord = pred[:, self.num_anc:, :, :]

        lab_anc = label[:, :self.num_anc, :, :]
        lab_coord = label[:, self.num_anc:, :, :]

        #pred_anc = pred_anc * (nd.sum(nd.sum(lab_anc, axis=3, keepdims=True), axis=2, keepdims=True) > 0)
        #pred_coord = pred_coord * (nd.sum(nd.sum(lab_coord, axis=3, keepdims=True), axis=2, keepdims=True) > 0)
        pred_anc = pred_anc * (nd.sum(lab_anc, axis=1, keepdims=True) > 0)
        pred_coord = pred_coord * (nd.sum(lab_coord, axis=1, keepdims=True) > 0)
        #pred_anc = nd.reshape(pred_anc, [1,-1])
        #lab_anc = nd.reshape(lab_anc, [1,-1])
        """ Here shall be onehot-like arrays """


        pred_anc = nd.sum(nd.sum(pred_anc, axis=3, keepdims=False), axis=2, keepdims=False)
        pred_coord = nd.sum(nd.sum(pred_coord, axis=3, keepdims=False), axis=2, keepdims=False)
        lab_anc = nd.sum(nd.sum(lab_anc, axis=3, keepdims=False), axis=2, keepdims=False)
        lab_coord = nd.sum(nd.sum(lab_coord, axis=3, keepdims=False), axis=2, keepdims=False)

        pred_coord = pred_coord[:, np.argmax(lab_anc[0].asnumpy(),0)*4 : np.argmax(lab_anc[0].asnumpy(),0)*4+4]
        lab_coord = lab_coord[:, np.argmax(lab_anc[0].asnumpy(),0)*4 : np.argmax(lab_anc[0].asnumpy(),0)*4+4]

        return pred_anc, pred_coord, lab_anc, lab_coord


    def hybrid_forward(self, F, ypred, ylabel):
        assert ypred.shape == ylabel.shape, 'Check output shapes'
        anc_pred, coord_pred, anc_real, coord_real = self.split_signals(ypred, ylabel)

        if np.random.rand() > 0.9:
            print(anc_pred, anc_real)
            print(coord_pred, coord_real)

        sq_loss = nd.mean(nd.square(coord_pred - coord_real))
        soft_loss = nd.mean(nd.softmax_cross_entropy(data=nd.softmax(anc_pred, axis=1), label=nd.argmax(anc_real, -1)))

        ans_loss = 1. * soft_loss + 1. * sq_loss

        return ans_loss


class MSE(mx.metric.MSE):
    def __init__(self, logdir, tag, name='MSE'):
        super(MSE, self).__init__(name = name)
        self.writer = tf.summary.FileWriter(logdir)
        self.tag = tag
        self.iter = 0

    def update(self, gt_coords, preds):

        for coord, pred_coord in zip(gt_coords, preds):
            #if pred_label.shape != label.shape:
            #    pred_label = mx.ndarray.argmax(pred_label, axis=self.axis)
            #pred_label = mx.ndarray.argmax(pred_label, axis=self.axis).asnumpy().astype('int32')
            #label = mx.ndarray.argmax(label, axis=self.axis).asnumpy().astype('int32')
            #print pred_label.shape, label.shape
            self.sum_metric += np.mean(((gt_coords[0] - preds[0]).asnumpy() ** 2).ravel())
            self.num_inst += gt_coords[0].asnumpy().shape[0]

        self.iter += 1

    def write_results(self, step, tag):

        mse = self.get()
        summary = tf.Summary()
        summary.value.add(tag=tag, simple_value=mse[1])#[1])
        self.writer.add_summary(summary, int(step))


