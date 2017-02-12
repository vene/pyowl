""" OWL vs LASSO on a known correlated design.

Reproduces figure 1 from Figueiredo and Nowak,
Ordered Weighted L1 Regularized Regression with Strongly
Correlated Covariates: Theoretical Aspects.
http://www.jmlr.org/proceedings/papers/v51/figueiredo16.pdf
"""

# Author: Vlad Niculae <vlad@vene.ro>
# License: BSD 3 clause


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from pyowl import OwlRegressor

n_samples = 10
n_features = 100

coef = np.zeros(n_features)
coef[20:30] = -1
coef[60:70] = 1
coef /= np.linalg.norm(coef)

rng = np.random.RandomState(1)
X = rng.randn(n_samples, n_features)
X[:, 20:30] = X[:, 20]
X[:, 60:70] = X[:, 20]
X += 0.001 * rng.randn(n_samples, n_features)
X /= np.linalg.norm(X, axis=0)
y = np.dot(X, coef)

plt.figure()

# ground truth:
plt.subplot(221)
plt.stem(np.arange(n_features), coef)
plt.title("True coefficients")

alpha = 0.0001
beta = 0.01  # only in OWL

# scikit-learn LASSO
plt.subplot(222)
lasso_skl = Lasso(alpha=alpha / (2 * n_samples), fit_intercept=False)
lasso_skl.fit(X, y)
plt.stem(np.arange(n_features), lasso_skl.coef_)
plt.title("LASSO coefficients (scikit-learn)")

# pyowl lasso
plt.subplot(223)
lasso_owl = OwlRegressor(weights=np.ones(n_features) * alpha)
lasso_owl.fit(X, y)
plt.stem(np.arange(n_features), lasso_owl.coef_)
plt.title("LASSO coefficients (pyowl)")

# pyowl lasso
plt.subplot(224)
oscar_owl = OwlRegressor(weights=(alpha, beta))
oscar_owl.fit(X, y)
plt.stem(np.arange(n_features), oscar_owl.coef_)
plt.title("OSCAR coefficients (pyowl)")

plt.tight_layout()
plt.savefig("toy_example.png")

