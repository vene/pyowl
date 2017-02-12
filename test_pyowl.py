# Author: Vlad Niculae <vlad@vene.ro>
# License: BSD 3 clause

import numpy as np
from numpy.testing import assert_array_almost_equal
from pyowl import prox_owl

rng = np.random.RandomState(0)

# cf. scikit-learn-contrib/lightning impl/penalty.py
def project_simplex(v, z=1):
    if np.sum(v) <= z:
        return v

    n_features = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - z
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    return w


# cf. scikit-learn-contrib/lightning impl/penalty.py
def project_l1_ball(v, z=1):
    return np.sign(v) * project_simplex(np.abs(v), z)


def prox_linf(v, alpha):
    # cf. Proximal Algorithms, Parikh & Boyd, eq. 6.8
    # dual ball B is the L1 ball

    p = project_l1_ball(v / alpha)
    return v - alpha * p



def test_prox_special_cases():
    for _ in range(20):
        v = rng.randn(10)
        alpha = rng.uniform(0.001, 1)

        # l1 proximal operator
        z_expected = np.maximum(0, v - alpha)
        z_expected -= np.maximum(0, -v - alpha)
        z_obtained = prox_owl(v, alpha * np.ones_like(v))

        assert_array_almost_equal(z_expected, z_obtained)

        # l_inf proximal operator
        z_expected = prox_linf(v, alpha)
        w = np.zeros_like(v)
        w[0] = alpha
        z_obtained = prox_owl(v, w)
        assert_array_almost_equal(z_expected, z_obtained)
