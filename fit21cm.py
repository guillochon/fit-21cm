"""Fit 21cm absorption profile."""
# from dynesty import DynamicNestedSampler
from dynesty import NestedSampler
import numpy as np
from collections import OrderedDict
from matplotlib import pyplot as plt


def foreground(nus, a):
    """Foreground."""
    return a[0] * (nus / nu_c) ** -2.5 + a[1] * (nus / nu_c) ** -2.5 * np.log(
        nus / nu_c) + a[2] * (nus / nu_c) ** -2.5 * np.log(
            nus / nu_c) ** 2 + a[3] * (nus / nu_c) ** -4.5 + a[4] * (
                nus / nu_c) ** -2


def log_like(x):
    """Log likelihood."""
    fg = foreground(xdata, x)

    diff = ydata - fg
    return np.sum(-0.5 * np.inner(diff, diff) / x[5] ** 2)


def ptform(u):
    """Map priors to physical units."""
    x = u.copy()
    for vi, v in enumerate(free_vars):
        x[vi] = free_vars[v][0] + (free_vars[v][1] - free_vars[v][0]) * u[vi]
    return x


datalen = 1000
mu = 1.0
sigma = 0.1
min_nu = 51.0
max_nu = 99.0
nu_c = (max_nu - min_nu) / 2.0

free_vars = OrderedDict((
    ('a0', (-10000, 10000)),
    ('a1', (-10000, 10000)),
    ('a2', (-10000, 10000)),
    ('a3', (-10000, 10000)),
    ('a4', (-10000, 10000)),
    ('sigma', (0, 100))
))

ndim = len(list(free_vars.keys()))

xdata = np.linspace(min_nu, max_nu, datalen)
ydata = np.random.normal(mu, sigma, datalen)

dsampler = NestedSampler(log_like, ptform, ndim, bound='single')
dsampler.run_nested()

res = dsampler.results

weights = np.exp(res['logwt'])
weights /= np.sum(weights)

plt.plot(xdata, ydata, color='black', lw=1.5)
for si, samp in enumerate(res['samples']):
    if weights[si] < 2.e-3:
        continue
    plt.plot(xdata, foreground(xdata, samp), color='blue', lw=0.5, alpha=0.5)
plt.show()
