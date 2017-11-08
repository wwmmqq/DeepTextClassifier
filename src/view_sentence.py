# coding: utf-8
from __future__ import print_function
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import tools


def to2d(fname="samples_vector"):
    x, y = tools.load_params("../log/{}.pkl".format(fname))
    print(x.shape)
    tsne = TSNE()
    x_2d = tsne.fit_transform(x)
    print(x_2d.shape)
    tools.save_params([x_2d, y], "../log/{}_2d.pkl".format(fname))
    return [x_2d, y]


def to2d_pca(fname="samples_vector"):
    x, y = tools.load_params("../log/{}.pkl".format(fname))
    print(x.shape)
    pca = PCA(n_components=2)
    x_2d = pca.fit_transform(x)
    print(x_2d.shape)
    tools.save_params([x_2d, y], "../log/pca_{}_2d.pkl".format(fname))
    return [x_2d, y]

in_name = "center_divide_samples_vector"
x2d, labels = to2d_pca(in_name)
# tools.load_params("../log/nn_center_soft_loss_samples_vector_2d.pkl")

print (x2d.shape)

colors = ["b", "g", "r", "c", "m", "y"]
for i in range(6):
    d = x2d[np.where(labels == i)[0]]
    print (d.shape)

    plt.scatter(d[:, 0], d[:, 1], s=1, color=colors[i], label="%s" % i)

plt.legend()
plt.show()
