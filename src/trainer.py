import logging
import os
import pprint
import time

import joblib
import numpy
import torch
import torch.nn.functional as F
import yaml
from data_iterator import TextIterator
from model import ENLI_Model

from src.utils import prepare_data, unrap,pred_acc

logger = logging.getLogger(__name__)

def adjust_learning_rate(optimizer, decay_rate=.5):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate


def train():
    numpy.random.seed(497727774)
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")

    #load dictionary
    config = yaml.load(open('config.yml'))['model']

    worddicts = joblib.load(config['vocab'])

    logging.debug(pprint.pformat(config))

    print('Loading data')

    train = TextIterator(config['train']['pre'],
                         config['train']['hyp'],
                         config['train']['lab'],
                         dict=worddicts,
                         batch_size=config['batch_size'])
    valid = TextIterator(config['dev']['pre'],
                         config['dev']['hyp'],
                         config['dev']['lab'],
                         dict=worddicts,
                         batch_size=config['batch_size'])
    test = TextIterator(config['test']['pre'],
                         config['test']['hyp'],
                         config['test']['lab'],
                         dict=worddicts,
                         batch_size=config['batch_size'])

    print('test encoding')
    for x1, x2, y in test:
        print(x1,x2,y)
        break

    print('build model')

    model = ENLI_Model(len(worddicts),config,worddicts)

    if config['reload'] and os.path.exists(config['saveto']):
        print('Reload parameters')
        model.load_state_dict(torch.load(config['saveto']))
        # load_params(saveto,params)

    use_gpu = config['use_gpu']
    if config['mode'] == 'test':
        model.eval()
        vres = pred_acc(model, prepare_data, valid, use_gpu)
        print('muti Valid accuracy', vres[0])
        print('bi Valid accuracy', vres[1])
        print('valid auc', vres[2])
        tres = pred_acc(model, prepare_data, test, use_gpu)
        print('muti test accuracy', tres[0])
        print('bi test accuracy', tres[1])
        print('test auc', tres[2])

    if config['use_gpu']:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config['gpu_id'])
        #torch.cuda.set_device(config['gpu_id'))
        model.cuda()

    lrate = config['lrate']
    optimizer = torch.optim.Adam(model.parameters(), lr=lrate, weight_decay=config['decay_c'])
    #optimizer = torch.optim.Adadelta(model.parameters(), eps=1e-7, rho=0.99)
    history_errs = []

    #reload history
    if config['reload'] and os.path.exists(config['saveto']):
        print('Reload history error')
        history_errs =joblib.load(config['history_error'])

    best_p = None
    bad_counter = 0

    uidx = 0
    estop = False
    valid_acc_record = []
    test_acc_record = []
    best_epoch_num = 0
    lr_change_list = []
    wait_counter = 0
    wait_N = 1
    for eidx in range(config['max_epochs']):
        n_samples = 0
        for x1, x2, y in train:
            model.train()
            optimizer.zero_grad()
            n_samples += len(x1)
            uidx += 1
            x1, x1_mask, x2, x2_mask, y = prepare_data(x1, x2, y,use_gpu)
            if x1 is None:
                print('Minibatch with zero sample under length ')
                uidx -= 1
                continue
            ud_start = time.time()
            probs = model(x1,x1_mask,x2,x2_mask)
            loss = F.cross_entropy(probs, y)
            loss.backward()
            #nn.utils.clip_grad_norm(model.parameters(), config['clip_c'])
            optimizer.step()

            ud = time.time() - ud_start
            cost = unrap(loss,use_gpu).data.numpy()[0]
            if numpy.isnan(cost) or numpy.isinf(cost):
                print('NaN detected')
                return None

            if numpy.mod(uidx, config['dispFreq']) == 0:
                logger.debug('Epoch {0} Update {1} Cost {2} UD {3}'.format(eidx, uidx, cost, ud))


            if numpy.mod(uidx, config['saveFreq']) == 0:
                print('Saving...')
                if best_p is not None:
                    save_model = best_p   #the best model
                else:
                    save_model = model
                joblib.dump(history_errs,config['history_error'])
                torch.save(save_model.state_dict(),config['saveto'])
                print('Done')

            if numpy.mod(uidx, config['validFreq']) == 0:
                model.eval()
                v_muti,v_binary,v_auc = pred_acc(model , prepare_data, valid,use_gpu)
                v_err = 1.0 - v_auc
                history_errs.append(v_err)
                t_muti, t_binary, t_auc = pred_acc(model , prepare_data, test,use_gpu)
                print('Valid muti: {} binary: {} auc: {}'.format(v_muti, v_binary, v_auc))
                print('Test muti: {} binary: {} auc: {}'.format(t_muti, t_binary, t_auc))
                print('lrate:', lrate)

                valid_acc_record.append(v_auc)
                test_acc_record.append(t_auc)

                if uidx == 0 or v_err <= numpy.array(history_errs).min():
                    best_p = model
                    best_epoch_num = eidx
                    wait_counter = 0

                if v_err > numpy.array(history_errs).min():
                    wait_counter += 1

                if wait_counter >= wait_N:
                    print('wait_counter max, need to half the lr')
                    bad_counter += 1
                    wait_counter = 0
                    print('bad_counter: ' + str(bad_counter))
                    adjust_learning_rate(optimizer,0.5)
                    lrate = lrate * 0.5
                    lr_change_list.append(eidx)
                    print('lrate change to: ' + str(lrate))


                if bad_counter > config['patience']:
                    print('Early Stop!')
                    estop = True
                    break

            if uidx >= config['finish_after']:
                print('Finishing after %d iterations!' % uidx)
        print('Seen %d samples'%n_samples)
        if estop:
            break
    if best_p is not None:
        torch.save(best_p.state_dict(),config['saveto'])

    with open('record.csv','w') as f:
        f.write(str(best_epoch_num) + '\n')
        f.write(','.join(map(str, lr_change_list)) + '\n')
        f.write(','.join(map(str, test_acc_record)) + '\n')
        f.write(','.join(map(str, valid_acc_record)) + '\n')

    model.eval()
    vres = pred_acc(model , prepare_data,valid,use_gpu)
    print('muti Valid accuracy', vres[0])
    print('bi Valid accuracy', vres[1])
    print('valid auc', vres[2])
    tres = pred_acc(model , prepare_data,test,use_gpu)
    print('muti test accuracy', tres[0])
    print('bi test accuracy', tres[1])
    print('test auc', tres[2])

    joblib.dump(history_errs,config['history_error'])
    logger.debug('Done')

if __name__ == '__main__':
    train()