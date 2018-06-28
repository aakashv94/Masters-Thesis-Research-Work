# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 10:37:29 2018

@author: Aakash
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import time

start = time.clock()
#----------------------------------------------------------------------------
'''
Code for finding out the holding time from exponential distribution using scale param.
given by theta Vector in the form theta = (theta_1, theta_2).
'''
def Hold(T, theta): #Defining function Hold which takes 2 args: T-Horizon, Theta
    c,x,h = 0,0,[]
    while c<=T:
        c_last=c
        h1 = np.random.exponential(scale = theta[x]) #generating rand num from exp.
        c+=h1
        h.append(min(h1, T-c_last))
        x = 1-x
    return h # returns the Holding time in given units of T

#----------------------------------------------------------------------------

p = 0.1 
p_m = p/(1-p)

#Following are the scale parameters for drwaing rand no. from exp. dist.

theta_1 = np.arange(10.0,101.5,5.0) 
theta_2 =9*theta_1
length = len(theta_1)

#Time steps considering 250(d) * 60(h) * 6(m) to be the trading time.

dt = 1.0/(250*360) #Length of each partition of time horizon.
T = 1.0/dt #Time Horizon in minute unit.
#-----------------------------------------------------------------------------

'''
Now for generating the list of Holding times for considered thetas we use 
Holding_Time function created above and create an array of Holding_time 
(denoted by H_t) as below
'''
H_t = [list(Hold(T,[theta_1[i],theta_2[i]])) for i in range(length)] #in minute unit

X = [] #Defining state variable to store chain for corresponding theta values.
for j1 in range(0,length):
    x, X1 = 0, [] 
    for j2 in range(0, len(H_t[j1])):
        X1.append(x)  #Will give chain for particular theta (seq. of consecutive 0's and 1's.)
        x = 1-x
    X.append(X1) #This will return array for all thetas considered.

mu_bar = 1.7
mu0 = -1.0
mu1 = (mu_bar - (p*mu0))/(1-p)
mu = (mu0, mu1) #Drift values
sigma = (0.01, 0.01) #Volatility values

#----------------------------------------------------------------------------

#This is the code for generating random numbers from Normal Distribution using
#Box-Muller Transform.

def generate_normal(mu, variance): #Defining a function which takes args: mean(mu) & Variance
    sigma = np.sqrt(variance)
    U = np.random.uniform(0,1)
    V = np.random.uniform(0,1)
    X1 = (np.sqrt(-2 * np.log(1- U)))*(math.cos(2*(np.pi)*V))#Output: rand numb. from Std. Normal
    return (mu + sigma*X1) #Output: Normal Random Number with given mean and variance


#---------- Code for Markov Modulated Geometric Brownian Motion ---------------

def MMGBM(S0, Hold_t, state, Mu, Sigma):
    dt = 1.0/(250*360)
    sdt = 1 #time increments (in min unit)
    S =[S0]
    m=0
    a = S[0]
    while m<=len(state)-1:
        H = Hold_t[m]
        H1 = int(H)
        dt_1 = (H-H1)*dt
        for t in np.arange(sdt,H+1,sdt):
            a = a*np.exp((Mu[m] - 0.5*(Sigma[m]**2))*dt + Sigma[m]*(generate_normal(0, abs(dt))))
            S.append(a)    
        if (H-H1)>0:
            a = a*np.exp((Mu[m] - 0.5*(Sigma[m]**2))*dt_1 + Sigma[m]*(generate_normal(0, abs(dt_1))))
            S.append(a) 
        m+=1
    return S
    
#-------------------------------Duration Code---------------------------------#
def dur(est_arr,p): #Function to find out the squeezed duration of drift
                     #taking 2 args: list of estimators $mu_{j} and p %.
    d = []
    flag = 1
    for j1 in range(0,len(est_arr)):
        if est_arr[j1]*flag<p*flag:
            d.append(j1)
            flag = -flag
    durations = np.diff(d)[::2]
    return durations #Output: returns the squeezed duration for array of estimators

#dura = []
d_mean = []
d_sd = []

#Finding out the squeeze durations for given list of theta paremeters and 
#then computing duration mean and std. deviation for the thetas.


#-----------------------------------------------------------------------------#
#Code for computing Output i.e. simulated stock prices data for diff. thetas


n=20 #This is the window size for durations

for i1 in range(0,length):
    ret = []
    est = []
    X_theta = X[i1] # Considering chain for particular theta (X_theta) out of array of chains (X).
    Ht = H_t[i1] # consecutive holding times in minute unit
    Mu = [mu[i] for i in X_theta] # consecutive drifts in minute unit
    Sigma = [sigma[j] for j in X_theta] # consecutive vols in minute unit
    S = MMGBM(5, Ht, X_theta, Mu, Sigma) #Stores the stock Prices values using MMGBM fn.
    ret = [(S[i]-S[i-1])/S[i-1] for i in range(1,len(S))]    
    length1 = len(ret)
    for i2 in range(1,length1-n):
        est.append((1/dt)*(1/n)*sum(ret[i2:i2+n]))
    m_est = est
    c_dur = dur(m_est,0)
    cbar = np.mean(c_dur)
    n1 = len(c_dur)
    d_mean.append(cbar)
    ss = sum((x-cbar)**2 for x in c_dur)
    d_sd.append((ss/n1)**0.5)    
    

'''
Plotting the values of theta_1 vs d_mean and d_sd.
'''

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.set_xlim(min(theta_1)-0.5,max(theta_1)+0.5)
ax1.set_ylim(min(d_mean)-1,max(d_mean)+1)
plt.title(r'$\theta_{1}$ vs mean (squeezed duration)')
plt.xlabel(r'$\theta_{1}$')
plt.ylabel('mean(squeezed duration)')
plt.plot(theta_1, d_mean, 'ro')
#
fig2 = plt.figure()
ax1 = fig2.add_subplot(111)
ax1.set_xlim(min(theta_1)-0.5,max(theta_1)+0.5)
ax1.set_ylim(min(d_sd)-0.5,max(d_sd)+0.5)
plt.title(r'$\theta_{1}$ vs std (squeezed duration)')
plt.xlabel(r'$\theta_{1}$')
plt.ylabel('std(squeezed duration)')
plt.plot(theta_1, d_sd, 'bo')
print("--- %.3f seconds ---" % (time.clock() - start))