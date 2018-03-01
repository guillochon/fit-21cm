"""Fit 21cm absorption profile."""
import os
from collections import OrderedDict
from csv import reader

import numpy as np
from matplotlib import pyplot as plt
from corner import corner

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
    fg = b[0] * (nus / nu_c) ** (-2.5 + b[1] + b[2] * np.log(
        nus / nu_c)) * np.exp(-b[3] * (nus / nu_c) ** -2) + b[4] * (
            nus / nu_c) ** -2
    cut = 50000.0
    mincut = 0.1
    fg[fg > cut] = cut * (1.0 + np.log(fg[fg > cut] / cut))
    fg[fg < -cut] = -cut * (1.0 + np.log(-fg[fg < -cut] / cut))
    minind = np.logical_and(fg < mincut, fg > 0)
    fg[minind] = mincut / (
        1.0 - np.log(fg[minind] / mincut))
    minind = np.logical_and(fg > -mincut, fg < 0)
    fg[minind] = -mincut / (
        1.0 - np.log(-fg[minind] / mincut))
    return fg


def line_profile(nus, b):
    """21cm line function from Bowman 2018."""
    a = b[5]
    nu0 = b[6]
    w = b[7]
    tau = b[8]
    bigb = 4.0 * (nus - nu0) ** 2 / w ** 2 * np.log(
        -1.0 / tau * np.log((1.0 + np.exp(-tau)) / 2.0))
    return -a * (1.0 - np.exp(-tau * np.exp(bigb))) / (1.0 - np.exp(-tau))


def log_like(x):
    """Log likelihood."""
    fg = foreground2(xdata, x)

    diff = ydata - fg
    return -0.5 * (np.sum(np.inner(diff, diff) / x[-1] ** 2) + len(
        diff) * np.log(x[-1] ** 2))


def ptform(u):
    """Map priors to physical units."""
    x = u.copy()
    for vi, v in enumerate(free_vars):
        x[vi] = free_vars[v][0] + (free_vars[v][1] - free_vars[v][0]) * u[vi]
    return x


arange = 5000

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

free_vars = OrderedDict((
    # ('a0', (-arange, arange)),
    # ('a1', (-arange, arange)),
    # ('a2', (-arange, arange)),
    # ('a3', (-arange, arange)),
    # ('a4', (-arange, arange)),
    # ('a0', (0, 5000)),
    # ('a1', (0, 1200)),
    # ('a2', (-2000, 0)),
    # ('a3', (0, 1500)),
    # ('a4', (-1000, 500)),
    ('b0', (0, 4000)),
    ('b1', (-2, 2)),
    ('b2', (-2, 2)),
    ('b3', (-2, 2)),
    ('b4', (-1500, 1500)),
    ('A', (0, 10)),
    ('nu0', (min_nu, max_nu)),
    ('w', (0, 200)),
    ('tau', (0, 100)),
    ('sigma', (0, 10))
))

ndim = len(list(free_vars.keys()))

nu_c = (max_nu + min_nu) / 2.0

dsampler = NestedSampler(
    log_like, ptform, ndim, sample='rwalk', nlive=500)
dsampler.run_nested(dlogz=0.001)

res = dsampler.results

weights = res['logwt']
weights -= np.max(weights)

# plt.plot(xdata, ydata, color='black', lw=1.5)
corner_weights = []
corner_vars = []
rms = []
for si, samp in enumerate(res['samples']):
    if weights[si] < -7:
        continue
    corner_weights.append(np.exp(weights[si]))
    corner_vars.append(samp)
    diff = ydata - (foreground2(xdata, samp) + line_profile(xdata, samp))
    rms.append(np.sqrt(np.var(diff)))
    if weights[si] > -2:
        plt.plot(xdata, diff, color='blue', lw=0.5, alpha=0.5)

print('RMS: {}'.format(np.median(rms)))

plt.savefig("21cm.pdf")

plt.clf()

try:
    corner(corner_vars, weights=corner_weights, labels=list(free_vars.keys()),
           range=[0.99 for x in range(ndim)])
except AssertionError as e:
    print(repr(e))
plt.savefig("corner.pdf")

# plt.show()
