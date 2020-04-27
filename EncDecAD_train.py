import argparse
import chainer
from chainer import training
from chainer.training import extensions
import pickle
from chainer.iterators import SerialIterator
from LSTM_func import EncDecAD
from LSTM_func import LSTM_MSE
from LSTM_func import LSTM_Iterator
from LSTM_func import LSTMUpdater
from LSTM_func import MyEvaluator

def main():
    parser = argparse.ArgumentParser(description='Chainer LSTM Network')
    parser.add_argument('--batchsize', '-b', type=int, default=8,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=80,
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
    parser.add_argument('--resume', '-r', default="result/snapshot_iter_1119", type=str,
                        help='Resume the training from snapshot')
    parser.add_argument('--autoload', action='store_true',
                        help='Automatically load trainer snapshots in case'
                             ' of preemption or other temporary system failure')
    parser.add_argument('--unit', '-u', type=int, default=20,
                        help='Number of units')
    # parser.add_argument('--noplot', dest='plot', action='store_false', help='Disable PlotReport extension')
    parser.add_argument('--plot', type=bool, default=True, help='Disable PlotReport extension')
    parser.add_argument("--train_path", type=str, default="train.sav")
    parser.add_argument("--test_path", type=str, default="test.sav")
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
    with open(args.train_path, "rb") as f:
        train = pickle.load(f)
    with open(args.test_path, "rb") as f:
        test = pickle.load(f)

    print("train:", len(train), "test:", len(test))

    train_iter = SerialIterator(train, args.batchsize)
    test_iter = SerialIterator(test, args.batchsize, repeat=False)

    # Set up a neural network to train
    net = EncDecAD(50, 3500)
    model = LSTM_MSE(net)
    model.to_device(device)
    device.use()

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # Set up a trainer
    updater = LSTMUpdater(train_iter, optimizer, device=device)
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
        ['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

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
