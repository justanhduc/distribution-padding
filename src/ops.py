from theano import gradient as G
from theano.gpuarray.dnn import dnn_pool as pool
from theano import tensor as T
import neuralnet as nn
import numpy as np


def mean_interp_pad(x, padding):
    padding = (padding, padding) if isinstance(padding, int) else tuple(padding)
    size = tuple(np.array(padding) * 2 + 1)
    resize = ((x.shape[2] + 2 * padding[0], x.shape[2] - 2 * padding[0]),
              (x.shape[3] + 2 * padding[1], x.shape[3] - 2 * padding[1]))
    y = pool(x, size, (1, 1), mode='average_exc_pad')
    z = G.disconnected_grad(nn.utils.frac_bilinear_upsampling(y, resize))
    _, _, h, w = z.shape
    return T.set_subtensor(z[:, :, padding[0]:h - padding[0], padding[1]:w - padding[1]], x)


def mean_ref_pad(x, padding):
    padding = (padding, padding) if isinstance(padding, int) else tuple(padding)
    size = tuple(np.array(padding) * 2 + 1)
    resize = ((x.shape[2], x.shape[2] - 2 * padding[0]),
              (x.shape[3], x.shape[3] - 2 * padding[1]))
    y = pool(x, size, (1, 1), mode='average_exc_pad')
    z = nn.utils.frac_bilinear_upsampling(y, resize)
    z = G.disconnected_grad(nn.utils.reflection_pad(z, padding))
    _, _, h, w = z.shape
    return T.set_subtensor(z[:, :, padding[0]:h - padding[0], padding[1]:w - padding[1]], x)
