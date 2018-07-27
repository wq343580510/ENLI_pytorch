# coding=utf-8
import numpy
from torch.autograd import Variable
import torch
import os
import joblib
from utils import *
from sklearn import metrics
from model import ENLI_Model
from data_iterator import TextIterator



def pred_acc_ensemble(model_lst, prepare_data, iterator,use_gpu = False):
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
        print(n_done)
        x1, x1_mask, x2, x2_mask, y = prepare_data(x1, x2, y,use_gpu)
        y = unrap(y,use_gpu).data.numpy()
        probs = numpy.zeros((y.size,4))
        for model in model_lst:
            prob = unrap(model(x1, x1_mask, x2, x2_mask),use_gpu).data.numpy()
            probs += prob
        probs /= len(model_lst)
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


def test():
    # load dictionary
    config = {'use_gpu': True,
              'hidden_units': 400,
              'vocab': './train_data/enli.dict',
              'word_dim': 200,
              'gpu_id': 14,
              'dropout': 0.2,
              'n_word': 42394,
              'batch_size': 32
              }

    worddicts = joblib.load(config['vocab'])


    print('Loading data')

    prefix = './train_data/bs_new.utf8'
    test = TextIterator('{}.query'.format(prefix),
                        '{}.title'.format(prefix),
                        '{}.label'.format(prefix),
                        dict=worddicts,
                        batch_size=config['batch_size'])

    model_file_lst = ['/home/disk0/wangqi38/pytorch-final/save_files/enli_0.2_400_shuffle.pkl',
                      '/home/disk0/wangqi38/pytorch-final/save_files/enli_0.3_400.pkl']
    model_lst = []
    print('load models')
    if config['use_gpu']:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config['gpu_id'])
    for model_name in model_file_lst:
        model = ENLI_Model(config)
        if config['use_gpu']:
            model.load_state_dict(torch.load(model_name))
        else:
            model.load_state_dict(torch.load(model_name, map_location={'cuda:0': 'cpu'}))
        model.eval()
        model_lst.append(model)

    use_gpu = config['use_gpu']

    tres = pred_acc_ensemble(model_lst, prepare_data, test, use_gpu)
    print('muti test accuracy', tres[0])
    print('bi test accuracy', tres[1])
    print('test auc', tres[2])



    print('finish')


test()
