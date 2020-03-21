import warnings
import chainer
from chainer import function
import chainer.links as L
from chainer import training
from chainer import reporter
from chainer.training import extensions
import cupy as cp
import copy
from chainer import reporter as reporter_module
import time


# Network definition
# LSTM_autoencoder
class EncDecAD(chainer.Chain):
    def __init__(self, n_in=2, n_units=4, train=True):
        super(EncDecAD, self).__init__()
        with self.init_scope():
            self.encLSTM = L.LSTM(in_size=n_in, out_size=n_units, lateral_init=chainer.initializers.Normal(scale=0.01))
            self.decLSTM = L.LSTM(in_size=n_in, out_size=n_units, lateral_init=chainer.initializers.Normal(scale=0.01))
            self.l1 = L.Linear(in_size=n_units, out_size=n_in, initialW=chainer.initializers.Normal(scale=0.01))
            self.l2 = L.Swish(beta_shape=n_in)
            self.train = train

    def __call__(self, X_sequence):
        y_sequence = []
        seq_len = len(X_sequence)
        self.reset_state()

        # Encoding
        for i in range(seq_len):
            self.encLSTM(X_sequence[i])

        # Decoding
        self.decLSTM.h = self.encLSTM.h
        if self.train is True:
            for i in range(seq_len):
                y = self.l2(self.l1(self.decLSTM.h))
                y_sequence.append(y)
                self.decLSTM(X_sequence[-i])
        else:
            for i in range(seq_len):
                y = self.l2(self.l1(self.decLSTM.h))
                y_sequence.append(y)
                self.decLSTM(y)

        y_sequence.reverse()
        return y_sequence

    def reset_state(self):
        self.encLSTM.reset_state()
        self.decLSTM.reset_state()


class LSTM_MSE(L.Classifier):
    def __init__(self, predictor):
        super(LSTM_MSE, self).__init__(predictor)

    def __call__(self, x):
        batch_size = len(x)
        self.loss = 0

        for i in range(batch_size):
            shape = x[i].shape
            pred = self.predictor(x[i].reshape([shape[0], 1, shape[1]]))
            self.loss += self.mean_squeared_error(pred, x[i])

        # 各lossの平均を取る
        self.loss /= batch_size
        # reporter に loss の値を渡す
        reporter.report({'loss': self.loss}, self)

        return self.loss

    def mean_squeared_error(self, x1, x2):  # x1 = pred(list of Variable), x2 = x[i](ndarray)
        mse = 0
        seq_len = x2.shape[0]
        dim = x2.shape[1]

        for s in range(seq_len):
            for d in range(dim):
                mse += (x1[s][0][d] - x2[s][d]) ** 2

        return mse


class LSTMUpdater(training.StandardUpdater):
    def __init__(self, data_iter, optimizer, device=None):
        super(LSTMUpdater, self).__init__(data_iter, optimizer, device=None)
        self.device = device

    def update_core(self):
        data_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')

        x_batch = data_iter.__next__()

        optimizer.target.cleargrads()
        loss = optimizer.target(x_batch)  # cp:18sec np:13sec
        loss.backward()  # cp:25sec np:86sec
        loss.unchain_backward()
        optimizer.update()


class LSTM_Iterator(chainer.dataset.Iterator):
    def __init__(self, dataset, batch_size=100, seq_len=20, support_len=10, repeat=True):
        self.seq_length = seq_len
        self.support_len = support_len
        self.nsamples = len(dataset)
        self.batch_size = batch_size
        self.repeat = repeat

        start = time.time()
        self.x = cp.array([dataset[i:i + self.seq_length] for i in range(self.nsamples - self.seq_length)],
                          dtype="float32")
        print("Cupy allocation time:", time.time() - start)

        self.epoch = 0
        self.iteration = 0
        self.loop = 0
        self.is_new_epoch = False

    def __next__(self):
        if not self.repeat and self.iteration >= 1:
            raise StopIteration

        self.iteration += 1
        if self.repeat:
            self.offsets = cp.random.randint(0, self.nsamples - self.seq_length - 1, size=self.batch_size)
        else:
            self.offsets = cp.arange(0, self.nsamples - self.seq_length - 1)

        x = self.get_data()
        self.epoch = int((self.iteration * self.batch_size) // self.nsamples)

        return x

    def get_data(self):
        x = [self.x[os] for os in self.offsets]

        return x

    @property
    def epoch_detail(self):
        return self.epoch

    def reset(self):
        self.epoch = 0
        self.iteration = 0
        self.loop = 0


class MyEvaluator(extensions.Evaluator):
    def evaluate(self):
        iterator = self._iterators['main']
        eval_func = self._targets['main']

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            warnings.warn(
                'This iterator does not have the reset method. Evaluator '
                'copies the iterator instead of resetting. This behavior is '
                'deprecated. Please implement the reset method.',
                DeprecationWarning)
            it = copy.copy(iterator)

        summary = reporter_module.DictSummary()

        for x_batch, t_batch in it:
            observation = {}
            with reporter_module.report_scope(observation):
                with function.no_backprop_mode():
                    eval_func(x_batch, t_batch)

            summary.add(observation)

        return summary.compute_mean()
