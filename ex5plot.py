import matplotlib.pyplot as plt
import numpy as np


def sec1():
    fig, ax = plt.subplots()
    ax.axvspan(1.5, 2.5, alpha=0.5, color='red')
    ax.axvspan(0, 1.5, alpha=0.5, color='green')
    ax.axvspan(2.5, 4, alpha=0.5, color='green')

    plt.scatter([1], [3], c='b', marker='+', s=100)
    plt.scatter([3], [3], c='b', marker='+', s=100)
    plt.scatter([2], [1], c='b', marker='_', s=100)
    plt.scatter([2], [4], c='r', marker='o', s=100)
    plt.ylabel('v2')
    plt.xlabel('v1')

    plt.show()


def sec2():
    x = np.arange(0.0, 3, 0.1)
    y = x
    fig, ax = plt.subplots()
    ax.fill_between(x, y, 0, color='red', alpha=0.5)
    ax.fill_between(x, y, 3, color='green', alpha=0.5)

    plt.scatter([1], [2], c='b', marker='+', s=100)
    plt.scatter([0], [1], c='b', marker='+', s=100)
    plt.scatter([2], [1], c='b', marker='_', s=100)
    plt.scatter([1], [0], c='b', marker='_', s=100)
    plt.scatter([0.5], [0.7], c='r', marker='o', s=100)
    plt.ylabel('v2')
    plt.xlabel('v1')

    plt.show()


def sec3():
    fig, ax = plt.subplots()
    ax.axvspan(0, 4, alpha=0.5, color='green')
    ax.axvspan(0.8, 1.2, alpha=0.5, color='red')
    ax.axvspan(2.8, 3.2, alpha=0.5, color='red')

    plt.scatter([1], [1], c='b', marker='_', s=100)
    plt.scatter([2], [1], c='b', marker='+', s=100)
    plt.scatter([3], [1], c='b', marker='_', s=100)
    plt.scatter([3.5], [1], c='g', marker='o', s=100)
    plt.ylabel('v2')
    plt.xlabel('v1')

    plt.show()


def sec4():
    fig, ax = plt.subplots()
    ax.axvspan(0, 2, alpha=0.5, color='green')
    ax.axvspan(2, 4, alpha=0.5, color='red')

    plt.scatter([3], [1], c='b', marker='_', s=100)
    plt.scatter([1], [1], c='b', marker='+', s=100)
    plt.ylabel('v2')
    plt.xlabel('v1')

    plt.show()

sec4()