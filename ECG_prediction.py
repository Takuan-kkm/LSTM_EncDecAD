import argparse
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer.iterators import SerialIterator
import pickle
from LSTM_func import EncDecAD
from LSTM_func import LSTM_MSE
from LSTM_func import LSTM_Iterator
from LSTM_func import LSTMUpdater
from LSTM_func import MyEvaluator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import cupy as cp
import sys
import matplotlib.pyplot as plt
import os
import pickle


def main():
    parser = argparse.ArgumentParser(description='Chainer LSTM Network')
    parser.add_argument('--batchsize', '-b', type=int, default=64,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=15,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--frequency', '-f', type=int, default=-1,
                        help='Frequency of taking a snapshot')
    parser.add_argument('--device', '-d', type=str, default='0',
                        help='Device specifier. Either ChainerX device '
                             'specifier or an integer. If non-negative integer, '
                             'CuPy arrays with specified device id are used. If '
                             'negative integer, NumPy arrays are used')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', type=str,
                        help='Resume the training from snapshot')
    parser.add_argument('--autoload', action='store_true',
                        help='Automatically load trainer snapshots in case'
                             ' of preemption or other temporary system failure')
    parser.add_argument('--unit', '-u', type=int, default=20,
                        help='Number of units')
    # parser.add_argument('--noplot', dest='plot', action='store_false', help='Disable PlotReport extension')
    parser.add_argument('--plot', type=bool, default=True, help='Disable PlotReport extension')
    parser.add_argument("--train_path", type=str, default="UCRArchive_2018/ECG5000/ECG5000_TRAIN.tsv")
    parser.add_argument("--test_path", type=str, default="UCRArchive_2018/ECG5000/ECG5000_TEST.tsv")
    group = parser.add_argument_group('deprecated arguments')
    group.add_argument('--gpu', '-g', dest='device',
                       type=int, nargs='?', const=0,
                       help='GPU ID (negative value indicates CPU)')
    args = parser.parse_args()

    device = chainer.get_device(args.device)

    print('Device: {}'.format(device))
    print('# unit: {}'.format(args.unit))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    chainer.backends.cuda.set_max_workspace_size(512 * 1024 * 1024)
    chainer.config.autotune = True

    # Load dataset
    df = pd.read_table(args.test_path, header=None, nrows=1000, usecols=[x for x in range(1, 141)])
    print("DATASET:", args.test_path)

    # 行ごとに切り出してXに格納
    X = []
    for i in range(1000):
        X.append([[j] for j in list(df.iloc[i])])
    # test,trainデータに分割
    X_train, X_test = train_test_split(X, test_size=0.2)

    train = cp.array(X_train, dtype="float32")
    test = cp.array(X_test, dtype="float32")
    print("train:", len(train), "test:", len(test))

    # train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    train_iter = SerialIterator(train, 1)
    test_iter = SerialIterator(test, args.batchsize, repeat=False)
    print("Iterator initialized.")

    next = test_iter.__next__()
    # 元波形をプロット
    x = [i[0] for i in next[0]]
    plt.xlim([0, 140])
    plt.ylim([-5, 2.5])
    plt.plot(x)
    plt.show()

    # モデルのロード
    with open("result_ECG/model.pkl", "rb") as f:
        model = pickle.load(f)

    # モデルによる復元
    shape = next[0].shape
    with chainer.using_config('train', True):
        print("train:", model.train)
        pred = model(next[0].reshape([shape[0], 1, shape[1]]))

    # train = False
    # 復元波形のプロット
    pred_plt = [x[0][0].array for x in pred]
    # print(pred_plt[0])
    plt.xlim([0, 140])
    plt.ylim([-5, 2.5])
    plt.plot(pred_plt)
    plt.show()

    with chainer.using_config('train', False):
        model.train = False
        print("train:", model.train)
        pred = model(next[0].reshape([shape[0], 1, shape[1]]))

    # 復元波形のプロット
    pred_plt = [x[0][0].array for x in pred]
    # print(pred_plt[0])
    plt.xlim([0, 140])
    plt.ylim([-5, 2.5])
    plt.plot(pred_plt)
    plt.show()

    print("loss:", LSTM_MSE.mean_squeared_error(LSTM_MSE, pred, next[0]))


if __name__ == '__main__':
    main()
