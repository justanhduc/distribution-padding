import argparse

parser = argparse.ArgumentParser('Distribution padding')
parser.add_argument('path', type=str, help='path to the CIFAR10 dataset')
parser.add_argument('-a', '--arch', type=str, default='ResNet34MeanInterpPadding', help='network architecture')
parser.add_argument('-b', '--bs', type=int, default=128, help='batch size')
parser.add_argument('-l', '--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('-r', '--l2_coeff', type=float, default=5e-4, help='l2 regularization factor')
parser.add_argument('-e', '--n_epochs', type=int, default=100, help='number of epochs')
parser.add_argument('-p', '--print_freq', type=int, default=1000, help='displaying frequency')
parser.add_argument('-v', '--valid_freq', type=int, default=1000, help='validation frequency')
parser.add_argument('-g', '--gpu', type=int, default=1, help='gpu number')
args = parser.parse_args()

import os
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

import neuralnet as nn
import numpy as np
from theano import tensor as T
from functools import partial

from networks import (
    VGG19MeanInterpPadding,
    VGG19,
    VGG19MeanRefPadding,
    ResNet34MeanInterpPadding,
    ResNet34,
    ResNet34MeanRefPadding,
    ResNet34RefPadding
)
from data_loader import CIFAR10


path = args.path
image_shape = (3, 48, 48)
X_train, y_train, X_test, y_test = nn.data_loader.load_dataset(path, True, new_shape=image_shape[1:])
architecture = args.arch
bs = args.bs
l2_coeff = args.l2_coeff
lr = args.lr
n_epochs = args.n_epochs
print_freq = args.print_freq
valid_freq = args.valid_freq

transforms = [nn.transforms.RandomCrop(image_shape[1:], padding=4),
              nn.transforms.RandomHorizontalFlip()]
nets = {'VGG19': VGG19, 'ResNet34': ResNet34,
        'VGG19RefPadding': partial(VGG19, border_mode='ref', name='VGG19RefPadding'),
        'VGG19PartialConvPadding': partial(VGG19, border_mode='partial', name='VGG19PartialConvPadding'),
        'ResNet34MeanInterpPadding': ResNet34MeanInterpPadding, 'VGG19MeanInterpPadding': VGG19MeanInterpPadding,
        'ResNet34MeanRefPadding': ResNet34MeanRefPadding, 'VGG19MeanRefPadding': VGG19MeanRefPadding,
        'ResNet34RefPadding': ResNet34RefPadding}


def train():
    X = T.tensor4('images')
    y = T.lvector('labels')

    X_ = nn.placeholder((bs,) + image_shape, name='images_plhd')
    y_ = nn.placeholder((bs,), dtype='int64', name='labels_plhd')

    net = nets[architecture]((None,) + image_shape)
    nn.set_training_on()
    updates, losses, grad_norms = net.get_updates(X, y, l2_coeff, lr)
    train_net = nn.function([], list(losses.values()), updates=updates,
                            givens={X: X_, y: y_}, name='train net')

    nn.set_training_off()
    err, losses = net.get_accuracy(X, y)
    valid_net = nn.function([], [err, losses['loss']], givens={X: X_, y: y_}, name='validate net')

    train_data = CIFAR10((X_train, y_train), (X_, y_), bs, n_epochs, training=True, shuffle=True, augmentation=transforms)
    valid_data = CIFAR10((X_test, y_test), (X_, y_), bs, 1, training=False, shuffle=False)
    mon = nn.Monitor(model_name=architecture, print_freq=print_freq)
    for it in train_data:
        with mon:
            losses_ = train_net()
            if np.any(np.isnan(losses_)) or np.any(np.isinf(losses_)):
                raise ValueError('NAN loss!')

            for j, k in enumerate(losses.keys()):
                mon.plot(k, losses_[j])

            if it % valid_freq == 0:
                mean_res = np.mean([valid_net() for _ in valid_data], 0)
                mon.plot('validation error', mean_res[0])
                mon.plot('validation loss', mean_res[1])
                mon.dump(nn.utils.shared2numpy(net.params), '%s.npy' % architecture, 5)
    print('Training finished!')


if 'main' in __name__:
    train()
