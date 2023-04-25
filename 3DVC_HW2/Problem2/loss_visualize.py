import matplotlib.pyplot as plt
import numpy as np
import json
import os


def loss_visualize(filename):
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

    # print(index)
    # print(loss)
    # return

    # plot loss
    plt.figure()
    plt.plot(index, loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    # plt.ylim(0, 2.0)
    plt.xticks(np.arange(0, index[-1] + 1, index[-1] // 10))
    plt.show()
    plt.clf()

    plt.savefig(filename.replace('.json', '.png'))


if __name__ == '__main__':
    dir = r'D:\OldNew\3DVC\image2pcd\outputs-LeakyReLU-step-HDLoss-noisy'
    # loss_visualize(os.path.join(dir, 'training_loss.json'))
    loss_visualize(os.path.join(dir, 'eval_loss.json'))
