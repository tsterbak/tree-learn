'''
Created on 11.08.2016

@author: Tobias
'''
from itertools import product

import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from tree import logistic_tree, decision_tree_c45, simple_tree, rbf_tree

# Loading some example data
# iris = datasets.load_iris()
# X = iris.data[:, [0, 2]]
# y = iris.target
noisy_moons = datasets.make_moons(n_samples=1000, noise=.1)
X, y = noisy_moons

# Training classifiers
clf1 = DecisionTreeClassifier(max_depth=6)
clf2 = logistic_tree(max_depth=2, max_features=None, min_samples=100)
clf3 = rbf_tree(max_depth=2, max_features=None, min_samples=100)
clf4 = decision_tree_c45(max_depth=6, max_features=None)

clf1.fit(X, y)
clf2.fit(X, y)
clf3.fit(X, y)
clf4.fit(X, y)

# Plotting decision regions
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

f, axarr = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(10, 8))

for idx, clf, tt in zip(product([0, 1], [0, 1]),
                        [clf1, clf2, clf3, clf4],
                        ['Sklearn Decision Tree', 'Logistic Tree',
                         'RBF Tree', 'Decision Tree']):

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.4)
    axarr[idx[0], idx[1]].scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
    axarr[idx[0], idx[1]].set_title(tt)

plt.show()