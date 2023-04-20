import os
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
def _tanh(x):
    return (math.exp(x) - math.exp(-x))/(math.exp(x) + math.exp(-x))


def _guss(x, sigma=0.2):
    return math.exp((-(x**2)/sigma))

if __name__ == '__main__':

    img_files = os.listdir("/workspace/CornerNet/data_CityPerson/coco/images/val/")
    x = np.arange(0,2,0.01)
    y = []
    for t in x:
        y1 = _guss(_tanh(t))
        y.append(y1)
    plt.plot(x,y)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.savefig("/workspace/CornerAff/visual_result/func.png")

    x = np.arange(0, 2, 0.01)
    y = []
    for t in x:
        y1 = _guss(_tanh(t))
        y.append(y1)
    plt.plot(x, y)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.savefig("/workspace/CornerAff/visual_result/func.png")



