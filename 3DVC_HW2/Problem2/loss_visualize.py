import matplotlib.pyplot as plt
import numpy as np
import json
import os


def loss_visualize(filename):
    # load loss
    with open(filename, 'r') as f:
        loss = json.load(f)
    # plot loss
    plt.figure()
    plt.plot(loss['train'], label='train')
    plt.plot(loss['val'], label='val')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(os.path.join('outputs', 'loss.png'))
    plt.show()
    plt.close()
    # plot loss
    plt.figure()
    plt.plot(loss['train'], label='train')
    plt.plot(loss['val'], label='val')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.yscale('log')
    plt.savefig(os.path.join('outputs', 'loss_log.png'))
    plt.show()
    plt.close()
    # plot loss
    plt.figure()
    plt.plot(loss['train'], label='train')
    plt.plot(loss['val'], label='val')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.yscale('log')
    plt.ylim(1e-3, 1e0)
    plt.savefig(os.path.join('outputs', 'loss_log_zoom.png'))
    plt.show()
    plt.close()