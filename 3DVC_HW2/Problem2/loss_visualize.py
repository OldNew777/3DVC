import matplotlib.pyplot as plt
import numpy as np
import json
import os


def load_loss(filename):
    # load loss
    with open(filename, 'r') as f:
        loss = json.load(f)

    if type(loss) is list:
        index = np.arange(len(loss))
        loss = np.array(loss)
    elif type(loss) is dict:
        index = np.array(list(loss.keys()))
        index = index.astype(int)
        loss = np.array(list(loss.values()))
    else:
        raise ValueError('Unknown loss format')
    return index, loss


def loss_visualize(filename):
    index, loss = load_loss(filename)

    # print(index)
    # print(loss)
    # return

    # plot loss
    plt.figure()
    plt.plot(index, loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    # plt.ylim(0, 0.11)
    plt.ylim(0, 2)
    # plt.xticks(np.arange(0, index[-1] + 1, index[-1] // 10))

    plt.savefig(filename.replace('.json', '.png'))
    plt.show()
    plt.clf()


def scatter_picture(filename):
    index, loss = load_loss(filename)

    pass


if __name__ == '__main__':
    dir = r'D:\OldNew\3DVC\image2pcd\outputs-LeakyReLU-step-HDLoss-clean'
    loss_visualize(os.path.join(dir, 'training_loss.json'))
    # loss_visualize(os.path.join(dir, 'eval_loss.json'))
