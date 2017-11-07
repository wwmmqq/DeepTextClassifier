# coding: utf-8
from __future__ import print_function
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import tools


def to2d():
    x, y = tools.load_params("../log/samples_vector.pkl")
    print(x.shape)
    tsne = TSNE()
    x_2d = tsne.fit_transform(x)
    print(x_2d.shape)
    tools.save_params([x_2d, y], "../log/samples_vector_2d.pkl")


x2d, labels = tools.load_params("../log/samples_vector_2d.pkl")

print (x2d.shape)

colors = ["b", "g", "r", "c", "m", "y"]
for i in range(6):
    d = x2d[np.where(labels == i)[0]]
    print (d.shape)

    plt.scatter(d[:, 0], d[:, 1], s=1, color=colors[i], label="%s" % i)

plt.legend()
plt.show()
