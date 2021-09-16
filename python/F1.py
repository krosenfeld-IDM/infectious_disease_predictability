'''
Looking at:

https://www.rdocumentation.org/packages/pdc/versions/1.0.3/topics/entropyHeuristic

https://github.com/amath-idm/psychics_vs_models/blob/master/dynamics/pe_search.py

https://github.com/brandmaier/pdc/blob/master/R/entropyHeuristic.R
'''

import os
import ordpy
import numpy as np
import pandas as pd
from collections import OrderedDict

import matplotlib.pyplot as plt
plt.rcParams["font.size"] = 12.
font_family = ['DejaVu Sans', 'Garamond', 'Proxima Nova'][2]
plt.rcParams["font.family"] = font_family
plt.rcParams["axes.formatter.use_mathtext"] = True
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["figure.dpi"] = 150
def axes_setup(axes):
    axes.spines["left"].set_position(("axes",-0.025))
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    return

import sys
sys.path.append(os.path.join('..'))
import utils

meas = pd.read_csv(os.path.join('..',  'Data', 'MEASLES_Cases_1909-2001_20150923120449.csv'))
for col in meas.columns:
    try:
        meas[col] = meas[col].astype(float)
    except:
        pass
print(meas.head())

# construct time series
meas_TX = meas[meas['YEAR'] < 1965]['TEXAS']
meas_TX_filt  = utils.filt_lead_trail_NA(meas_TX.values)

data = {}
data['Measles-Texas'] = meas_TX_filt['x']
years = meas.iloc[meas_TX_filt['first']:meas_TX_filt['last']+1]['YEAR'].values
time = meas.iloc[meas_TX_filt['first']:meas_TX_filt['last']+1]['YEAR'].values +  (meas.iloc[meas_TX_filt['first']:meas_TX_filt['last']+1]['WEEK'].values - 1)/52
n = len(data['Measles-Texas'])
data['noise-full'] = np.random.randn(n)
data['noiseMissing-missing'] = data['noise-full'].copy()
data['noiseMissing-missing'][np.isnan(data['Measles-Texas'])] = np.nan
data['sine-noise'] = np.sin(np.arange(1, n+1)) + np.random.randn(n)*0.001
data = pd.DataFrame(data)


# analysis
window = 52
results = OrderedDict()
plot_data = {}
for i, col in enumerate(data.columns):
    x_i = data.iloc[:, i]

    fit_i = utils.full_length_pred_window(x_i.values, 0, len(x_i), window=window, d_1=2, d_2=5, match_pdc=False)

    results.update({col: fit_i})
    plot_data.update({col: x_i})

# figure
xs = 5.5
idy = 2
xm = [0.9, 0.5]
ym = [0.5, 0.25, 0.5]
xcm = np.cumsum(xm)
ycm = np.cumsum(ym)
idx = xs - np.sum(xm)
ys = idy*2 + np.sum(ym)
fig_nrm = np.array([xs, ys, xs, ys])

cols = ("#7A8A83", "#ECBBAD", "#9D5241", "#334B58", "#7A8A83")

fig = plt.figure(figsize=(xs, ys))

# ax 1: weekly cases
rect = np.array([xcm[0], ycm[0], idx, idy])
ax1 = plt.axes(rect/fig_nrm)
ax1.plot(time, data['Measles-Texas'])
ax1.set_ylabel('Weekly Cases')

# ax 2: permutation entropy
rect = np.array([xcm[0], ycm[1]+idy, idx, idy])
ax2 = plt.axes(rect/fig_nrm)
for i, (k, v) in enumerate(results.items()):
    ax2.plot(v['results']['raw_perm_entropy'], color=cols[i], label=k)
ax2.legend(fontsize=8)
ax2.set_ylabel('Permutation Entropy')
ax2.set_ylim(0.75, 1.0)


plt.savefig(os.path.join('Figures', 'F1.png'))