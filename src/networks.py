from collections import OrderedDict

import neuralnet as nn
from theano import tensor as T

from ops import mean_interp_pad, mean_ref_pad


class Conv2DMeanInterpPaddingLayer(nn.Conv2DLayer):
    def __init__(self, input_shape, num_filters, filter_size, init=nn.HeNormal(gain='relu'), activation='relu',
                 stride=1, no_bias=False, layer_name='Conv2D Mean Interp Padding', **kwargs):
        super(Conv2DMeanInterpPaddingLayer, self).__init__(input_shape, num_filters, filter_size, init, activation=activation,
                                                           border_mode='half', stride=stride, no_bias=no_bias,
                                                           layer_name=layer_name, **kwargs)

    def get_output(self, input, *args, **kwargs):
        padding = self.filter_shape[-1] // 2
        x = mean_interp_pad(input, padding)
        y = T.nnet.conv2d(x, self.W, subsample=self.subsample, **self.kwargs)
        if not self.no_bias:
            y += self.b.dimshuffle('x', 0, 'x', 'x')
        return self.activation(y, **kwargs)


class Conv2DMeanRefPaddingLayer(nn.Conv2DLayer):
    def __init__(self, input_shape, num_filters, filter_size, init=nn.HeNormal(gain='relu'), activation='relu',
                 stride=1, no_bias=False, layer_name='Conv2D Mean Ref Padding', **kwargs):
        super(Conv2DMeanRefPaddingLayer, self).__init__(input_shape, num_filters, filter_size, init,
                                                        activation=activation, border_mode='half', stride=stride,
                                                        no_bias=no_bias, layer_name=layer_name, **kwargs)

    def get_output(self, input, *args, **kwargs):
        padding = self.filter_shape[-1] // 2
        x = mean_ref_pad(input, padding)
        y = T.nnet.conv2d(x, self.W, subsample=self.subsample, **self.kwargs)
        if not self.no_bias:
            y += self.b.dimshuffle('x', 0, 'x', 'x')
        return self.activation(y, **kwargs)


class Net:
    def get_cost(self, x, y, l2_coeff):
        preds = self(x)
        cost = nn.multinoulli_cross_entropy(preds, y)
        l2 = nn.l2_reg(self.params)
        total = cost + l2 * l2_coeff
        losses = OrderedDict([('loss', cost), ('l2', l2), ('total', total)])
        return losses

    def get_updates(self, x, y, l2_coeff, lr):
        losses = self.get_cost(x, y, l2_coeff)
        updates, _, grads = nn.adam(losses['total'], self.trainable, lr)
        grad_norms = dict(
            [(self.trainable[idx].name.replace('/', '_'), nn.utils.p_norm(grad)) for idx, grad in enumerate(grads)])
        return updates, losses, grad_norms

    def get_accuracy(self, x, y):
        preds = self(x)
        losses = self.get_cost(x, y, 0.)
        error = nn.mean_classification_error(preds, y)
        return error, losses

    def get_feature_maps(self, x, layer):
        return self[:layer](x)


class VGG19MeanInterpPadding(nn.Sequential, Net):
    def __init__(self, input_shape, num_classes=10, name='vgg19 mean interp padding'):
        super(VGG19MeanInterpPadding, self).__init__(input_shape=input_shape, layer_name=name)
        self.append(Conv2DMeanInterpPaddingLayer(self.output_shape, 64, 3, activation=None, layer_name=name + '/conv1'))
        self.append(nn.BatchNormLayer(self.output_shape, name + '/bn1'))
        self.append(Conv2DMeanInterpPaddingLayer(self.output_shape, 64, 3, activation=None, layer_name=name + '/conv2'))
        self.append(nn.BatchNormLayer(self.output_shape, name + '/bn2', activation=None))
        self.append(nn.MaxPoolingLayer(self.output_shape, (2, 2), layer_name=name + '/maxpool0'))
        self.append(nn.ActivationLayer(self.output_shape, 'relu', name + '/relu2'))

        self.append(Conv2DMeanInterpPaddingLayer(self.output_shape, 128, 3, activation=None, layer_name=name + '/conv3'))
        self.append(nn.BatchNormLayer(self.output_shape, name + '/bn3'))
        self.append(Conv2DMeanInterpPaddingLayer(self.output_shape, 128, 3, activation=None, layer_name=name + '/conv4'))
        self.append(nn.BatchNormLayer(self.output_shape, name + '/bn4', activation=None))
        self.append(nn.MaxPoolingLayer(self.output_shape, (2, 2), layer_name=name + '/maxpool1'))
        self.append(nn.ActivationLayer(self.output_shape, 'relu', name + '/relu4'))

        self.append(Conv2DMeanInterpPaddingLayer(self.output_shape, 256, 3, activation=None, layer_name=name + '/conv5'))
        self.append(nn.BatchNormLayer(self.output_shape, name + '/bn5'))
        self.append(Conv2DMeanInterpPaddingLayer(self.output_shape, 256, 3, activation=None, layer_name=name + '/conv6'))
        self.append(nn.BatchNormLayer(self.output_shape, name + '/bn6'))
        self.append(Conv2DMeanInterpPaddingLayer(self.output_shape, 256, 3, activation=None, layer_name=name + '/conv7'))
        self.append(nn.BatchNormLayer(self.output_shape, name + '/bn7'))
        self.append(Conv2DMeanInterpPaddingLayer(self.output_shape, 256, 3, activation=None, layer_name=name + '/conv7_1'))
        self.append(nn.BatchNormLayer(self.output_shape, name + '/bn7_1', activation=None))
        self.append(nn.MaxPoolingLayer(self.output_shape, (2, 2), layer_name=name + '/maxpool2'))
        self.append(nn.ActivationLayer(self.output_shape, 'relu', name + '/relu8'))

        self.append(Conv2DMeanInterpPaddingLayer(self.output_shape, 512, 3, activation=None, layer_name=name + '/conv8'))
        self.append(nn.BatchNormLayer(self.output_shape, name + '/bn8'))
        self.append(Conv2DMeanInterpPaddingLayer(self.output_shape, 512, 3, activation=None, layer_name=name + '/conv9'))
        self.append(nn.BatchNormLayer(self.output_shape, name + '/bn9'))
        self.append(Conv2DMeanInterpPaddingLayer(self.output_shape, 512, 3, activation=None, layer_name=name + '/conv10'))
        self.append(nn.BatchNormLayer(self.output_shape, name + '/bn10'))
        self.append(Conv2DMeanInterpPaddingLayer(self.output_shape, 512, 3, activation=None, layer_name=name + '/conv10_1'))
        self.append(nn.BatchNormLayer(self.output_shape, name + '/bn10_1', activation=None))
        self.append(nn.MaxPoolingLayer(self.output_shape, (2, 2), layer_name=name + '/maxpool3'))
        self.append(nn.ActivationLayer(self.output_shape, 'relu', name + '/relu11'))

        self.append(Conv2DMeanInterpPaddingLayer(self.output_shape, 512, 3, activation=None, layer_name=name + '/conv11'))
        self.append(nn.BatchNormLayer(self.output_shape, name + '/bn11'))
        self.append(Conv2DMeanInterpPaddingLayer(self.output_shape, 512, 3, activation=None, layer_name=name + '/conv12'))
        self.append(nn.BatchNormLayer(self.output_shape, name + '/bn12'))
        self.append(Conv2DMeanInterpPaddingLayer(self.output_shape, 512, 3, activation=None, layer_name=name + '/conv13'))
        self.append(nn.BatchNormLayer(self.output_shape, name + '/bn13'))
        self.append(Conv2DMeanInterpPaddingLayer(self.output_shape, 512, 3, activation=None, layer_name=name + '/conv13_1'))
        self.append(nn.BatchNormLayer(self.output_shape, name + '/bn13_1', activation=None))
        self.append(nn.MaxPoolingLayer(self.output_shape, (2, 2), layer_name=name + '/maxpool4'))
        self.append(nn.ActivationLayer(self.output_shape, 'relu', name + '/relu14'))

        self.append(nn.SoftmaxLayer(self.output_shape, num_classes, name + '/softmax'))


class VGG19MeanRefPadding(nn.Sequential, Net):
    def __init__(self, input_shape, num_classes=10, name='vgg19 mean ref padding'):
        super(VGG19MeanRefPadding, self).__init__(input_shape=input_shape, layer_name=name)
        self.append(Conv2DMeanRefPaddingLayer(self.output_shape, 64, 3, activation=None, layer_name=name + '/conv1'))
        self.append(nn.BatchNormLayer(self.output_shape, name + '/bn1'))
        self.append(Conv2DMeanRefPaddingLayer(self.output_shape, 64, 3, activation=None, layer_name=name + '/conv2'))
        self.append(nn.BatchNormLayer(self.output_shape, name + '/bn2', activation=None))
        self.append(nn.MaxPoolingLayer(self.output_shape, (2, 2), layer_name=name + '/maxpool0'))
        self.append(nn.ActivationLayer(self.output_shape, 'relu', name + '/relu2'))

        self.append(Conv2DMeanRefPaddingLayer(self.output_shape, 128, 3, activation=None, layer_name=name + '/conv3'))
        self.append(nn.BatchNormLayer(self.output_shape, name + '/bn3'))
        self.append(Conv2DMeanRefPaddingLayer(self.output_shape, 128, 3, activation=None, layer_name=name + '/conv4'))
        self.append(nn.BatchNormLayer(self.output_shape, name + '/bn4', activation=None))
        self.append(nn.MaxPoolingLayer(self.output_shape, (2, 2), layer_name=name + '/maxpool1'))
        self.append(nn.ActivationLayer(self.output_shape, 'relu', name + '/relu4'))

        self.append(Conv2DMeanRefPaddingLayer(self.output_shape, 256, 3, activation=None, layer_name=name + '/conv5'))
        self.append(nn.BatchNormLayer(self.output_shape, name + '/bn5'))
        self.append(Conv2DMeanRefPaddingLayer(self.output_shape, 256, 3, activation=None, layer_name=name + '/conv6'))
        self.append(nn.BatchNormLayer(self.output_shape, name + '/bn6'))
        self.append(Conv2DMeanRefPaddingLayer(self.output_shape, 256, 3, activation=None, layer_name=name + '/conv7'))
        self.append(nn.BatchNormLayer(self.output_shape, name + '/bn7'))
        self.append(Conv2DMeanRefPaddingLayer(self.output_shape, 256, 3, activation=None, layer_name=name + '/conv7_1'))
        self.append(nn.BatchNormLayer(self.output_shape, name + '/bn7_1', activation=None))
        self.append(nn.MaxPoolingLayer(self.output_shape, (2, 2), layer_name=name + '/maxpool2'))
        self.append(nn.ActivationLayer(self.output_shape, 'relu', name + '/relu8'))

        self.append(Conv2DMeanRefPaddingLayer(self.output_shape, 512, 3, activation=None, layer_name=name + '/conv8'))
        self.append(nn.BatchNormLayer(self.output_shape, name + '/bn8'))
        self.append(Conv2DMeanRefPaddingLayer(self.output_shape, 512, 3, activation=None, layer_name=name + '/conv9'))
        self.append(nn.BatchNormLayer(self.output_shape, name + '/bn9'))
        self.append(Conv2DMeanRefPaddingLayer(self.output_shape, 512, 3, activation=None, layer_name=name + '/conv10'))
        self.append(nn.BatchNormLayer(self.output_shape, name + '/bn10'))
        self.append(
            Conv2DMeanRefPaddingLayer(self.output_shape, 512, 3, activation=None, layer_name=name + '/conv10_1'))
        self.append(nn.BatchNormLayer(self.output_shape, name + '/bn10_1', activation=None))
        self.append(nn.MaxPoolingLayer(self.output_shape, (2, 2), layer_name=name + '/maxpool3'))
        self.append(nn.ActivationLayer(self.output_shape, 'relu', name + '/relu11'))

        self.append(Conv2DMeanRefPaddingLayer(self.output_shape, 512, 3, activation=None, layer_name=name + '/conv11'))
        self.append(nn.BatchNormLayer(self.output_shape, name + '/bn11'))
        self.append(Conv2DMeanRefPaddingLayer(self.output_shape, 512, 3, activation=None, layer_name=name + '/conv12'))
        self.append(nn.BatchNormLayer(self.output_shape, name + '/bn12'))
        self.append(Conv2DMeanRefPaddingLayer(self.output_shape, 512, 3, activation=None, layer_name=name + '/conv13'))
        self.append(nn.BatchNormLayer(self.output_shape, name + '/bn13'))
        self.append(
            Conv2DMeanRefPaddingLayer(self.output_shape, 512, 3, activation=None, layer_name=name + '/conv13_1'))
        self.append(nn.BatchNormLayer(self.output_shape, name + '/bn13_1', activation=None))
        self.append(nn.MaxPoolingLayer(self.output_shape, (2, 2), layer_name=name + '/maxpool4'))
        self.append(nn.ActivationLayer(self.output_shape, 'relu', name + '/relu14'))

        self.append(nn.SoftmaxLayer(self.output_shape, num_classes, name + '/softmax'))


class VGG19(nn.model_zoo.VGG19, Net):
    def __init__(self, input_shape, border_mode='half', num_classes=10, name='VGG19'):
        super(VGG19, self).__init__(input_shape, fc=False, bn=True, border_mode=border_mode, num_classes=num_classes,
                                    name=name)
        self.append(nn.MaxPoolingLayer(self.output_shape, (2, 2), layer_name=name + '/maxpool4'))
        self.append(nn.SoftmaxLayer(self.output_shape, num_classes, name + '/softmax'))


class ResNet34(nn.model_zoo.ResNet, Net):
    def __init__(self, input_shape, num_classes=10, name='ResNet34'):
        super().__init__(input_shape, nn.ResNetBlock, (3, 4, 6, 3), 64, num_classes=num_classes, name=name,
                         pooling=False)


def _build_dist_pad_conv_block(input_shape, num_filters, stride, activation, block_name, **kwargs):
    block = [Conv2DMeanInterpPaddingLayer(input_shape, num_filters, 3, stride=stride, layer_name=block_name + '/conv1',
                                          no_bias=False, activation='linear')]
    block.append(nn.BatchNormLayer(block[-1].output_shape, activation=activation, layer_name=block_name + '/conv1_bn'))

    block.append(
        Conv2DMeanInterpPaddingLayer(block[-1].output_shape, num_filters, 3, layer_name=block_name + '/conv2',
                                     no_bias=True, activation='linear'))
    block.append(nn.BatchNormLayer(block[-1].output_shape, layer_name=block_name + '/conv2_bn', activation='linear'))
    return block


def _build_mean_ref_pad_conv_block(input_shape, num_filters, stride, activation, block_name, **kwargs):
    block = [Conv2DMeanRefPaddingLayer(input_shape, num_filters, 3, stride=stride, layer_name=block_name + '/conv1',
                                       no_bias=False, activation='linear')]
    block.append(nn.BatchNormLayer(block[-1].output_shape, activation=activation, layer_name=block_name + '/conv1_bn'))

    block.append(
        Conv2DMeanRefPaddingLayer(block[-1].output_shape, num_filters, 3, layer_name=block_name + '/conv2',
                                  no_bias=True, activation='linear'))
    block.append(nn.BatchNormLayer(block[-1].output_shape, layer_name=block_name + '/conv2_bn', activation='linear'))
    return block


def _build_simple_conv_block(input_shape, num_filters, stride, border_mode, activation, block_name, **kwargs):
    block = [nn.Conv2DLayer(input_shape, num_filters, 3, stride=stride, border_mode=border_mode,
                            layer_name=block_name + '/conv1', no_bias=False, activation='linear')]
    block.append(nn.BatchNormLayer(block[-1].output_shape, activation=activation, layer_name=block_name + '/conv1_bn'))

    block.append(nn.Conv2DLayer(block[-1].output_shape, num_filters, 3, border_mode=border_mode, no_bias=True,
                                layer_name=block_name + '/conv2', activation='linear'))
    block.append(nn.BatchNormLayer(block[-1].output_shape, layer_name=block_name + '/conv2_bn', activation='linear'))
    return block


class ResNet34MeanInterpPadding(nn.model_zoo.ResNet, Net):
    def __init__(self, input_shape, num_classes=10, name='ResNet34 Mean Interp Padding'):
        super().__init__(input_shape, nn.ResNetBlock, (3, 4, 6, 3), 64, num_classes=num_classes, name=name,
                         main_branch=_build_dist_pad_conv_block, pooling=False)


class ResNet34MeanRefPadding(nn.model_zoo.ResNet, Net):
    def __init__(self, input_shape, num_classes=10, name='ResNet34 Mean Ref Padding'):
        super().__init__(input_shape, nn.ResNetBlock, (3, 4, 6, 3), 64, num_classes=num_classes, name=name,
                         main_branch=_build_mean_ref_pad_conv_block, pooling=False)


class ResNet34RefPadding(nn.model_zoo.ResNet, Net):
    def __init__(self, input_shape, num_classes=10, name='ResNet34 Ref Padding'):
        super().__init__(input_shape, nn.ResNetBlock, (3, 4, 6, 3), 64, num_classes=num_classes, name=name,
                         main_branch=_build_simple_conv_block, border_mode='ref', pooling=False)
