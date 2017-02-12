# Author: Vlad Niculae <vlad@vene.ro>
# License: BSD 3 clause

import numpy as np


def squared_loss(y_true, y_pred, return_derivative=False):
    diff = y_pred - y_true
    obj = 0.5 * np.dot(diff, diff)
    if return_derivative:
        return obj, diff
    else:
        return obj


def squared_hinge_loss(y_true, y_scores, return_derivative=False):
    # labels in (-1, 1)
    z = np.maximum(0, 1 - y_true * y_scores)
    obj = np.sum(z ** 2)

    if return_derivative:
        return obj, -2 * y_true * z
    else:
        return obj


def get_loss(name):
    losses = {'squared': squared_loss,
              'squared-hinge': squared_hinge_loss}
    return losses[name]
