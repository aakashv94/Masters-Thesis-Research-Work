# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 10:36:50 2017

@author: Aakash
"""

import numpy as np
import matplotlib.pyplot as plt
n = int(input("Enter number of steps: \n"))
x = np.zeros(n)
data_x = [0]
for i in range(1,n):
  data_x.append(i)
  u = np.random.uniform(0,1)
  if (u<0.5):
    x[i] = x[i-1] + 1
  else:
    x[i] = x[i-1] - 1
print(len(data_x), len(x))
plt.plot(data_x, x)
print(data_x, x)
