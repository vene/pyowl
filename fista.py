"""
Efficient implementation of FISTA.
"""

# Author: Mathieu Blondel
# License: BSD 3 clause
# based on https://gist.github.com/mblondel/5105786d740693a6996bcb8e482c7083

import numpy as np


def fista(sfunc, nsfunc, x0, max_iter=500, max_linesearch=20, eta=2.0, tol=1e-3,
          verbose=0):

    y = x0.copy()
    x = y
    L = 1.0
    t = 1.0

    for it in range(max_iter):
        f_old, grad = sfunc(y, True)

        for ls in range(max_linesearch):
            y_proj = nsfunc(y - grad / L, L)
            diff = (y_proj - y).ravel()
            sqdist = np.dot(diff, diff)
            dist = np.sqrt(sqdist)

            F = sfunc(y_proj)
            Q = f_old + np.dot(diff, grad.ravel()) + 0.5 * L * sqdist

            if F <= Q:
                break

            L *= eta

        if ls == max_linesearch - 1 and verbose:
            print("Line search did not converge.")

        if verbose:
            print("%d. %f" % (it + 1, dist))

        if dist <= tol:
            if verbose:
                print("Converged.")
            break

        x_next = y_proj
        t_next = (1 + np.sqrt(1 + 4 * t ** 2)) / 2.
        y = x_next + (t-1) / t_next * (x_next - x)
        t = t_next
        x = x_next

    return y_proj

