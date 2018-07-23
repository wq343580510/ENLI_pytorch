import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy

def fflayer_init(layer,ortho=True):
    w = norm_weight(layer.out_features,layer.in_features,0.01,ortho)
    layer.weight.data.copy_(torch.from_numpy(w))
    b = numpy.zeros((layer.out_features,)).astype('float32')
    layer.bias.data.copy_(torch.from_numpy(b))

def embeddings_init(Wemb):
    num_embeddings = Wemb.num_embeddings
    embedding_dim = Wemb.embedding_dim
    W = norm_weight(num_embeddings, embedding_dim)
    Wemb.weight.data.copy_(torch.from_numpy(W))

def bilstm_init(encoder):
    nin = encoder.input_size
    dim = encoder.hidden_size
    W = numpy.concatenate([norm_weight(nin, dim),
                           norm_weight(nin, dim),
                           norm_weight(nin, dim),
                           norm_weight(nin, dim)], axis=0)
    b = numpy.zeros((4 * dim,)).astype('float32')
    U = numpy.concatenate([ortho_weight(dim),
                           ortho_weight(dim),
                           ortho_weight(dim),
                           ortho_weight(dim)], axis=0)
    W_r = numpy.concatenate([norm_weight(nin, dim),
                           norm_weight(nin, dim),
                           norm_weight(nin, dim),
                           norm_weight(nin, dim)], axis=0)
    U_r = numpy.concatenate([ortho_weight(dim),
                           ortho_weight(dim),
                           ortho_weight(dim),
                           ortho_weight(dim)], axis=0)
    encoder.weight_ih_l0.data.copy_(torch.from_numpy(W))
    encoder.bias_ih_l0.data.copy_(torch.from_numpy(b))
    encoder.weight_hh_l0.data.copy_(torch.from_numpy(U))
    encoder.bias_hh_l0.data.copy_(torch.from_numpy(b))
    encoder.weight_ih_l0_reverse.data.copy_(torch.from_numpy(W_r))
    encoder.bias_ih_l0_reverse.data.copy_(torch.from_numpy(b))
    encoder.weight_hh_l0_reverse.data.copy_(torch.from_numpy(U_r))
    encoder.bias_hh_l0_reverse.data.copy_(torch.from_numpy(b))


# some utilities
def ortho_weight(ndim):
    """
    Random orthogonal weights
    Used by norm_weights(below), in which case, we
    are ensuring that the rows are orthogonal
    (i.e W = U \Sigma V, U has the same
    # of rows, V has the same # of cols)
    """
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype('float32')


def norm_weight(nin, nout=None, scale=0.01, ortho=True):
    """
    Random weights drawn from a Gaussian
    """
    if nout is None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight(nin)
    else:
        W = scale * numpy.random.randn(nin, nout)
    return W.astype('float32')