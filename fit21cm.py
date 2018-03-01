"""Fit 21cm absorption profile."""
import os
from collections import OrderedDict
from csv import reader

import numpy as np
from matplotlib import pyplot as plt

# from dynesty import DynamicNestedSampler
from dynesty import NestedSampler

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))


def foreground(nus, a):
    """Linear approximation to foreground."""
    return (a[0] * (nus / nu_c) ** -2.5) + ((
        a[1] * (nus / nu_c) ** -2.5) * np.log(
        nus / nu_c)) + ((a[2] * (nus / nu_c) ** -2.5) * np.log(
            nus / nu_c) ** 2) + (a[3] * (nus / nu_c) ** -4.5) + a[4] * (
                nus / nu_c) ** -2


def foreground2(nus, b):
    """Foreground."""
    return b[0] * (nus / nu_c) ** (-2.5 + b[1] + b[2] * np.log(
        nus / nu_c)) * np.exp(-b[3] * (nus / nu_c) ** -2) + b[4] * (
            nus / nu_c) ** -2


def log_like(x):
    """Log likelihood."""
    fg = foreground(xdata, x)

    diff = ydata - fg
    return -0.5 * (np.sum(np.inner(diff, diff) / x[5] ** 2) + len(
        diff) * np.log(x[5] ** 2))


def ptform(u):
    """Map priors to physical units."""
    x = u.copy()
    for vi, v in enumerate(free_vars):
        x[vi] = free_vars[v][0] + (free_vars[v][1] - free_vars[v][0]) * u[vi]
    return x


arange = 5000

free_vars = OrderedDict((
    # ('a0', (-arange, arange)),
    # ('a1', (-arange, arange)),
    # ('a2', (-arange, arange)),
    # ('a3', (-arange, arange)),
    # ('a4', (-arange, arange)),
    ('a0', (0, 5000)),
    ('a1', (100, 1500)),
    ('a2', (-2000, 0)),
    ('a3', (0, 1500)),
    ('a4', (-1000, 500)),
    ('sigma', (0, 10000))
))

ndim = len(list(free_vars.keys()))

# Real data.
sigr = reader(open(os.path.join(__location__, 'signal.csv'), newline=''))
xdata = []
ydata = []
for row in sigr:
    xdata.append(float(row[0]))
    ydata.append(float(row[1]))
xdata = np.array(xdata)
ydata = np.array(ydata)
min_nu, max_nu = min(xdata), max(xdata)

# Dummy data.
# datalen = 1000
# mu = 0.0
# sigma = 0.1
# min_nu = 51.0
# max_nu = 99.0
# xdata = np.linspace(min_nu, max_nu, datalen)
# ydata = np.random.normal(mu, sigma, datalen)

nu_c = (max_nu + min_nu) / 2.0

dsampler = NestedSampler(
    log_like, ptform, ndim, sample='rwalk')  # , print_progress=False)
dsampler.run_nested(maxiter=200000)

res = dsampler.results

weights = res['logwt']
weights -= np.max(weights)

# plt.plot(xdata, ydata, color='black', lw=1.5)
for si, samp in enumerate(res['samples']):
    if weights[si] < -10:
        continue
    plt.plot(xdata, ydata - foreground(xdata, samp),
             color='blue', lw=0.5, alpha=0.5)
plt.savefig("21cm.pdf")
# plt.show()
