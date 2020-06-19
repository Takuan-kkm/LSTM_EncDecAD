import pickle
import cupy as np
from LSTM_func import LSTM_MSE
from chainer.iterators import SerialIterator
import matplotlib.pyplot as plt


def main():
    # Load dataset
    with open("test.sav", "rb") as f:
        test = pickle.load(f)

    with open("confuse.sav", "rb") as f:
        confuse = pickle.load(f)

    # Load network
    with open("result_seq3sec/model.pkl", "rb") as f:
        net = pickle.load(f)

    model = LSTM_MSE(net)
    iter_test = SerialIterator(test[:-4], batch_size=1, repeat=False, shuffle=False)
    iter_confuse = SerialIterator(confuse[:-4], batch_size=1, repeat=False, shuffle=False)

    loss_test = []
    loss_confuse = []

    print("Computing loss of test N=", len(test))
    for seq in iter_test:
        print(iter_test.current_position, " ", end="")
        loss_test.append(model(seq).data)

    ltest = np.array(loss_test, dtype="float32")
    print("loss_test:############################\n", loss_test)
    print("\ntest.sav\nN:", ltest.shape[0], "\nmean:", np.mean(ltest), "\nstd:", np.std(ltest))

    print("Computing loss of confuse N=", len(confuse))
    for seq in iter_confuse:
        print(iter_confuse.current_position, " ", end="")
        loss_confuse.append(model(seq).data)
    lconfuse = np.array(loss_confuse, dtype="float32")
    print("loss_confuse:############################\n", loss_confuse)
    print("\nconfuse.sav\nN:", lconfuse.shape[0], "\nmean:", np.mean(lconfuse), "\nstd:", np.std(lconfuse))

    fig = plt.figure(figsize=[15, 5])
    ax = fig.add_subplot(111)
    ax.plot(ltest.tolist(), label="normal behavior/error")
    ax.set_ylim([0, 3])
    ax.set_xlim([0, len(test)])
    ax.set_xlabel("time[*0.5sec]")
    ax.set_ylabel("error")
    ax.legend()
    plt.show()

    fig = plt.figure(figsize=[15, 5])
    ax = fig.add_subplot(111)
    ax.plot(lconfuse.tolist(), label="confusing behavior/error")
    ax.set_ylim([0, 3])
    ax.set_xlim([0, len(confuse)])
    ax.set_xlabel("time[*0.5sec]")
    ax.set_ylabel("error")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()
