import argparse
import chainer
from chainer import training
from chainer.training import extensions
from chainer.training import StandardUpdater
from chainer.optimizer_hooks import WeightDecay
import pickle
import cupy as cp
from chainer.iterators import SerialIterator
from LSTM_func import EncDecAD
from LSTM_func import LSTM_MSE
from LSTM_func import LSTM_Iterator
from LSTM_func import LSTMUpdater
from LSTM_func import MyEvaluator
import os

chainer.config.autotune = True

SUBJECT_ID = "TEST_SHUTARO_1109"  # temporary variable
TEST_PATH = os.environ["ONEDRIVE"] + "/研究/2020実験データ/BIN/" + SUBJECT_ID + "_VALID.pkl"  # temporary variable
TRAIN_PATH = os.environ["ONEDRIVE"] + "/研究/2020実験データ/BIN/" + SUBJECT_ID + "_TRAIN.pkl"  # temporary variable

SUBJECTS = ["T1_1109", "S1_1112", "H1_1202", "E1_1203", "N1_1008", "A1_1217", "Y1_1217"]


def pkl_read(subjects, train=True):
    for index, s in enumerate(subjects):
        if train is True:
            path = os.environ["ONEDRIVE"] + "/研究/2020実験データ/CSV_BIN/" + s + "/" + s + "_TRAIN.pkl"
        else:
            path = os.environ["ONEDRIVE"] + "/研究/2020実験データ/CSV_BIN/" + s + "/" + s + "_VALID.pkl"

        if index == 0:
            with open(path, "rb") as f:
                print(path)
                data = pickle.load(f)
        else:
            with open(path, "rb") as f:
                print(path)
                data = cp.concatenate([data, pickle.load(f)])

    print(subjects)
    print("train:", train, "shape:", data.shape)

    return data


def main():
    parser = argparse.ArgumentParser(description='Chainer LSTM Network')
    parser.add_argument('--batchsize', '-b', type=int, default=256,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=150,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--frequency', '-f', type=int, default=150,
                        help='Frequency of taking a snapshot')
    parser.add_argument('--device', '-d', type=int, default=0,
                        help='Device specifier. Either ChainerX device '
                             'specifier or an integer. If non-negative integer, '
                             'CuPy arrays with specified device id are used. If '
                             'negative integer, NumPy arrays are used')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', type=str, default=None,
                        help='Resume the training from snapshot')
    #  default="result/snapshot_iter_1268"
    parser.add_argument('--autoload', action='store_true',
                        help='Automatically load trainer snapshots in case'
                             ' of preemption or other temporary system failure')
    parser.add_argument('--unit', '-u', type=int, default=3000,
                        help='Number of units')
    # parser.add_argument('--noplot', dest='plot', action='store_false', help='Disable PlotReport extension')
    parser.add_argument('--plot', type=bool, default=True, help='Disable PlotReport extension')
    parser.add_argument("--train_path", type=str, default=TRAIN_PATH)
    parser.add_argument("--test_path", type=str, default=TEST_PATH)
    group = parser.add_argument_group('deprecated arguments')
    group.add_argument('--gpu', '-g', dest='device',
                       type=int, nargs='?', const=0,
                       help='GPU ID (negative value indicates CPU)')
    args = parser.parse_args()

    device = chainer.get_device(args.device)
    print(device)

    print('Device: {}'.format(device))
    print('# unit: {}'.format(args.unit))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    chainer.backends.cuda.set_max_workspace_size(512 * 1024 * 1024)
    chainer.config.autotune = True

    # Load dataset
    train = pkl_read(subjects=SUBJECTS, train=True)
    valid = pkl_read(subjects=SUBJECTS, train=False)

    train_iter = SerialIterator(train, args.batchsize)
    test_iter = SerialIterator(valid, args.batchsize, repeat=False)

    # Set up a neural network to train
    net = EncDecAD(156, 1024)
    model = LSTM_MSE(net)
    model.to_device(device)
    device.use()

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # 重み減衰
    for param in net.params():
        if param.name != 'b':  # バイアス以外だったら
            param.update_rule.add_hook(WeightDecay(0.0001))  # 重み減衰を適用

    # Set up a trainer
    # updater = LSTMUpdater(train_iter, optimizer, device=device)
    updater = StandardUpdater(train_iter, optimizer, device=device)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(MyEvaluator(test_iter, model, device=device))

    # Take a snapshot for each specified epoch
    frequency = args.epoch if args.frequency == -1 else max(1, args.frequency)
    # Take a snapshot each ``frequency`` epoch, delete old stale
    # snapshots and automatically load from snapshot files if any
    # files are already resident at result directory.
    trainer.extend(extensions.snapshot(num_retain=1, autoload=args.autoload),
                   trigger=(frequency, 'epoch'))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport())

    # Save two plot images to the result dir
    if args.plot and extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(['main/loss', 'validation/main/loss'], 'epoch', file_name='loss.png')
        )

    # Print selected entries of the log to stdout
    # Here "main" refers to the target link of the "main" optimizer again, and
    # "validation" refers to the default name of the Evaluator extension.
    # Entries other than 'epoch' are reported by the Classifier link, called by
    # either the updater or the evaluator.
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss', 'elapsed_time']))

    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar())

    if args.resume is not None:
        # Resume from a snapshot (Note: this loaded model is to be
        # overwritten by --autoload option, autoloading snapshots, if
        # any snapshots exist in output directory)
        chainer.serializers.load_npz(args.resume, trainer)

    # Run the training
    trainer.run()

    # save the model
    with open("result/model.pkl", "wb") as f:
        pickle.dump(net, f)


if __name__ == '__main__':
    main()
