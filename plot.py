import argparse
import numpy as np
import matplotlib
import sys

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', dest='name', type=str, default='')
    parser.add_argument('--plot', dest='plot', type=bool, default=True)
    args = parser.parse_args()
    name = args.name
    training_episode = np.load('result/{}_training_ep.npy'.format(name))
    reward_mean= np.load('result/{}_reward_mean.npy'.format(name))
    reward_error= np.load('result/{}_reward_std.npy'.format(name))
    test_accuracy = np.load('result/{}_test_acc.npy'.format(name))

    print(training_episode)
    print(reward_mean)
    print(reward_error)
    print(test_accuracy)

    for i in range(len(reward_mean)):
        if reward_mean[i] >= 200:
            print(i)
            print(reward_mean[i])
            break

    max_acc = -float('Inf')
    max_idx = -1
    for i, score in enumerate(test_accuracy):
        if score > max_acc:
            max_acc = score
            max_idx = i
    print(max_idx)
    print(max_acc)
    print(test_accuracy[max_idx])


    if args.plot:
        # plt.errorbar(training_episode, reward_mean, reward_error)
        # plt.xlabel('Training Episode')
        # plt.ylabel('Cumulative Reward')
        # plt.savefig('plt/{}_reward.png'.format(name))
        plt.plot(training_episode, test_accuracy)
        plt.xlabel('Training Episode')
        plt.ylabel('test accuracy')
        plt.savefig('plt/{}_acc.png'.format(name))

        plt.gcf().clear()

        plt.plot(training_episode, reward_mean)
        plt.xlabel('Training Episode')
        plt.ylabel('test mean reward')
        plt.savefig('plt/{}_reward.png'.format(name))


if __name__ == "__main__":
    main(sys.argv)
