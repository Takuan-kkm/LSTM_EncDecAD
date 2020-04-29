import pickle
import cupy as np
from LSTM_func import LSTM_MSE
from chainer.iterators import SerialIterator


def main():
    # Load dataset
    with open("test.sav", "rb") as f:
        test = pickle.load(f)

    with open("confuse.sav", "rb") as f:
        confuse = pickle.load(f)

    # Load network
    with open("result/model.pkl", "rb") as f:
        net = pickle.load(f)

    model = LSTM_MSE(net)
    iter_test = SerialIterator(test, batch_size=1, repeat=False)
    iter_confuse = SerialIterator(confuse, batch_size=1, repeat=False)

    loss_test = []
    loss_confuse = []

    print("Computing loss of test N=", len(test))
    for seq in iter_test:
        print(iter_test.current_position, " ", end="")
        loss_test.append(model(seq).data)

    print("Computing loss of confuse N=", len(confuse))
    for seq in iter_confuse:
        print(iter_confuse.current_position, " ", end="")
        loss_confuse.append(model(seq).data)

    lconfuse = np.array(loss_confuse, dtype="float32")
    ltest = np.array(loss_test, dtype="float32")
    print("loss_test:############################\n", loss_test)
    print("loss_confuse:############################\n", loss_confuse)
    print("\ntest.sav\nN:", ltest.shape[0], "\nmean:", np.mean(ltest), "\nstd:", np.std(ltest))
    print("\nconfuse.sav\nN:", lconfuse.shape[0], "\nmean:", np.mean(lconfuse), "\nstd:", np.std(lconfuse))


if __name__ == "__main__":
    main()
