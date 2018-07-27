import logging
import os
import pprint
import numpy
import joblib
import torch
import yaml
from data_iterator import TextIterator
from model import ENLI_Model
import pandas as pd
from utils import prepare_data, unrap
from utils import pred_acc


def test():

    # load dictionary
    config = {'use_gpu':True,
              'hidden_units':400,
              'vocab': './train_data/enli.dict',
              'word_dim':200,
              'gpu_id':14,
              'dropout':0.2
              }

    worddicts = joblib.load(config['vocab'])

    logging.debug(pprint.pformat(config))

    print('Loading data')

    prefix = './train_data/bs_new.utf8'
    test = TextIterator('{}.query'.format(prefix),
                        '{}.title'.format(prefix),
                        '{}.label'.format(prefix),
                         dict=worddicts,
                         batch_size=config['batch_size'])


    print('build model')

    model = ENLI_Model(config, worddicts,load_pretrain=False)

    use_gpu = config['use_gpu']

    if config['use_gpu']:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config['gpu_id'])
        # torch.cuda.set_device(config['gpu_id'))
        model.cuda()


    print('Reload parameters')
    model.load_state_dict(torch.load('./rand/enli_0.2_400_shuffle.pkl',map_location={'cuda:0': 'cpu'}))
        # load_params(saveto,params)

    model.eval()
    tres = pred_acc(model, prepare_data, test, use_gpu)
    print('muti test accuracy', tres[0])
    print('bi test accuracy', tres[1])
    print('test auc', tres[2])

    results = []
    pred_labels = []
    model.eval()
    n_done = 0


    for x1, x2, y in test:
        n_done += len(x1)
        print(n_done)
        x1, x1_mask, x2, x2_mask, y = prepare_data(x1, x2, y,use_gpu)
        probs = unrap(model(x1, x1_mask, x2, x2_mask),use_gpu).data.numpy()
        pred_label = probs.argmax(axis=1)
        for prob in probs:
            results.append('-'.join([str('%.4f' % x) for x in prob]))
        for label in pred_label:
            pred_labels.append(label)
    #joblib.dump([results,pred_labels],'res.job')
    #data = joblib.load('res.job')
    print('see {} samples'.format(n_done))
    querys = open('{}.query'.format(prefix), 'r').readlines()
    titles = open('{}.title'.format(prefix), 'r').readlines()
    labels = open('{}.label'.format(prefix), 'r').readlines()
    df = pd.DataFrame({'query':querys,'title':titles,'label':labels,'pred_label':pred_labels,'probs':results})
    df.to_csv('{}_shuffle.csv'.format(prefix))
    print('finish')


test()