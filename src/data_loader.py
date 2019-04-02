import neuralnet as nn


class CIFAR10(nn.DataManager):
    def __init__(self, dataset, placeholders, bs, n_epochs, training=False, shuffle=False, **kwargs):
        super(CIFAR10, self).__init__(placeholders=placeholders, batch_size=bs, n_epochs=n_epochs, shuffle=shuffle,
                                      **kwargs)
        self.training = training
        self.dataset = dataset
        self.load_data()

    def load_data(self):
        self.data_size = self.dataset[0].shape[0]
