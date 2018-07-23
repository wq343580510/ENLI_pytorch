# coding=utf-8
import numpy
from torch.autograd import Variable
import torch
from sklearn import metrics

def rap(tensor,use_cuda=False):
    if use_cuda:
        return tensor.cuda()
    else:
        return tensor

def unrap(tensor,use_cuda=False):
    if use_cuda:
        return tensor.cpu()
    else:
        return tensor


def unrap(tensor,use_cuda=False):
    if use_cuda:
        return tensor.cpu()
    else:
        return tensor


def pred_acc(model, prepare_data, iterator,use_gpu):
    """
    Just compute the accuracy
    f_pred: Theano fct computing the prediction
    prepare_data: usual prepare_data for that dataset.
    """
    muti_valid_acc = 0
    bi_valid_acc = 0
    n_done = 0
    y_lst = []
    p_lst = []
    for x1, x2, y in iterator:
        n_done += len(x1)
        x1, x1_mask, x2, x2_mask, y = prepare_data(x1, x2, y,use_gpu)
        y = unrap(y,use_gpu).data.numpy()
        probs = unrap(model(x1, x1_mask, x2, x2_mask),use_gpu).data.numpy()
        label = probs.argmax(axis=1)
        bi_l = numpy.minimum(label,1)
        bi_y = numpy.minimum(y,1)
        muti_valid_acc += (label == y).sum()
        bi_valid_acc += (bi_l == bi_y).sum()
        for l in y:
            if l > 0:
                y_lst.append(1)
            else:
                y_lst.append(0)
        for p in probs:
            p_lst.append(p[1]+p[2]+p[3])
    auc = metrics.roc_auc_score(y_true=numpy.array(y_lst),y_score=numpy.array(p_lst))
    muti_valid_acc = 1.0 * muti_valid_acc / n_done
    bi_valid_acc = 1.0 * bi_valid_acc / n_done
    return muti_valid_acc,bi_valid_acc,auc

def prepare_data(seqs_x, seqs_y, labels,use_gpu = False, maxlen=None):
    lengths_x = [len(s) for s in seqs_x]
    lengths_y = [len(s) for s in seqs_y]
    if maxlen is not None:
        new_seqs_x = []
        new_seqs_y = []
        new_lengths_x = []
        new_lengths_y = []
        new_labels = []
        for l_x, s_x, l_y, s_y, l in zip(lengths_x, seqs_x, lengths_y, seqs_y, labels):
            if l_x < maxlen and l_y < maxlen:
                new_seqs_x.append(s_x)
                new_lengths_x.append(l_x)
                new_seqs_y.append(s_y)
                new_lengths_y.append(l_y)
                new_labels.append(l)
        lengths_x = new_lengths_x
        seqs_x = new_seqs_x
        lengths_y = new_lengths_y
        seqs_y = new_seqs_y
        labels = new_labels
        if len(lengths_x) < 1 or len(lengths_y) < 1:
            return None, None, None, None, None

    n_samples = len(seqs_x)
    maxlen_x = numpy.max(lengths_x)
    maxlen_y = numpy.max(lengths_y)

    x1 = numpy.zeros((maxlen_x, n_samples)).astype('int64')
    x2 = numpy.zeros((maxlen_y, n_samples)).astype('int64')
    x1_mask = numpy.zeros((maxlen_x, n_samples)).astype('float32')
    x2_mask = numpy.zeros((maxlen_y, n_samples)).astype('float32')
    y = numpy.zeros((n_samples,)).astype('int64')
    for idx, [s_x, s_y, ll] in enumerate(zip(seqs_x, seqs_y, labels)):
        x1[:lengths_x[idx], idx] = s_x
        x1_mask[:lengths_x[idx], idx] = 1.
        x2[:lengths_y[idx], idx] = s_y
        x2_mask[:lengths_y[idx], idx] = 1.
        y[idx] = ll
    x1 = rap(Variable(torch.from_numpy(numpy.transpose(x1))),use_gpu)
    x2 = rap(Variable(torch.from_numpy(numpy.transpose(x2))),use_gpu)
    x1_mask = rap(Variable(torch.from_numpy(numpy.transpose(x1_mask))),use_gpu)
    x2_mask = rap(Variable(torch.from_numpy(numpy.transpose(x2_mask))),use_gpu)
    y = rap(Variable(torch.from_numpy(y)),use_gpu)
    return x1, x1_mask, x2, x2_mask, y

