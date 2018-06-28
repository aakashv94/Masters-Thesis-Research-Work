# -*- coding: utf-8 -*-
'''
Created on Sat Dec 9 09:32:25 2017

@author: Aakash
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

'''
geometric brownian motion with drift!

    mu=drift factor
    sigma: volatility in %
    T: time span
    dt: lenght of steps
    S0: Stock Price in t=0
    W: Brownian Motion with Drift N[0,1]
'''

def geometric_brownian_motion(T = 1, N = 100, mu = 0.1, sigma = 0.01, S0 = 20):
    dt = float(T)/N
    t = np.linspace(0, T, N)
    W = np.random.standard_normal(size = N)
    W = np.cumsum(W)*np.sqrt(dt) ### standard brownian motion ###
    X = (mu-0.5*sigma**2)*t + sigma*W
    S = S0*np.exp(X) ### geometric brownian motion ###
    return S


dates = pd.date_range('2012-01-01', '2017-12-10')
T = (dates.max()-dates.min()).days / 365
N = dates.size
start_price = 100
y = pd.Series(
    geometric_brownian_motion(T, N, sigma=0.1, S0=start_price), index=dates)
y.plot()
plt.show()
