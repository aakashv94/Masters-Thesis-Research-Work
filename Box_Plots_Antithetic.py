#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 18:17:03 2018

@author: aakashverma
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import time
from collections import OrderedDict
import statistics as sc
start = time.clock()
#----------------------------------------------------------------------------

'''
Code for finding out the holding time from exponential distribution using scale param.
given by theta Vector in the form theta = (theta_1, theta_2).
'''
#---------------------Code : (Holding Time for 2 states[0,1])-----------------
def Hold(T, theta): #Defining function Hold that takes 2 args: T-Horizon, Theta
    c,x,h = 0,0,[]
    while c<=T:
        c_last=c
        h1 = np.random.exponential(scale = theta[x]) #generating rand num from exp.
        c+=h1
        h.append(min(h1, T-c_last))
        x = 1-x
    return h # returns the Holding time in given units of T

#----------------------------------------------------------------------------

#-------------- (Dynamic values for the MMGBM Problem investigation) ----------
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
Now for generating the list of Holding times for each of above given thetas we
use Holding_Time function created above and create an array of Holding_time 
(denoted by H_t) as below.
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
    return (mu + sigma*X1), (mu + sigma*(-X1)) #Output: Normal Random Number with given mean and variance


#---------- Code for (Markov Modulated Geometric Brownian Motion)--------------

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
            a = a*np.exp((Mu[m] - 0.5*(Sigma[m]**2))*dt + Sigma[m]*(generate_normal(0, abs(dt))[0]))
            S.append(a)    
        if (H-H1)>0:
            a = a*np.exp((Mu[m] - 0.5*(Sigma[m]**2))*dt_1 + Sigma[m]*(generate_normal(0, abs(dt_1))[0]))
            S.append(a) 
        m+=1
    return S  #Output is Simulated Stock Prices Data.

def MMGBM1(S0, Hold_t, state, Mu, Sigma):
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
            a = a*np.exp((Mu[m] - 0.5*(Sigma[m]**2))*dt + Sigma[m]*(generate_normal(0, abs(dt))[1]))
            S.append(a)    
        if (H-H1)>0:
            a = a*np.exp((Mu[m] - 0.5*(Sigma[m]**2))*dt_1 + Sigma[m]*(generate_normal(0, abs(dt_1))[1]))
            S.append(a) 
        m+=1
    return S  #Output is Simulated Stock Prices Data.

#------------------------------ (Duration Code) -------------------------------

def dur(est_arr,p): #Function to find out the squeezed duration of drift
                     #taking 2 args: list of estimators mu_{j} and p% cutoff.
    d = []
    flag = 1
    for j1 in range(0,len(est_arr)):
        if est_arr[j1]*flag<p*flag:
            d.append(j1)
            flag = -flag
    durations = np.diff(d)[::2]
    return durations #Output: returns the squeezed duration for array of estimators

#-----------------------------------------------------------------------------
#--Code for computing durations for (several iterations), dentoed by n_iter.---


t_d_mean = []
t_d_sd = []
t_d_mean1 = []
t_d_sd1 = []
n=20 #This is the window size for durations
n_iter = 25
stdv_antithetic = []
#---------------------------(n_iter Durations Code)-----------------------------
for i1 in range(0,length):
    X_theta = X[i1] # Considering chain for particular theta (X_theta) out of array of chains (X).
    Ht = H_t[i1] # consecutive holding times in minute unit
    Mu = [mu[i] for i in X_theta] # consecutive drifts in minute unit
    Sigma = [sigma[j] for j in X_theta] # consecutive vols in minute unit
    temp_mean = []
    temp_sd = []
    for j1 in range(0, n_iter):
        ret = []
        est = []
        S = MMGBM(5, Ht, X_theta, Mu, Sigma) #Stores the stock Prices values using MMGBM fn.
        ret = [(S[i]-S[i-1])/S[i-1] for i in range(1,len(S))]    
        length1 = len(ret)
        for i2 in range(1,length1-n):
            est.append((1/dt)*(1/n)*sum(ret[i2:i2+n]))
        m_est = est
        c_dur = dur(m_est,0)
        cbar = np.mean(c_dur)
        n1 = len(c_dur)
        ss = sum((x-cbar)**2 for x in c_dur)
        sd = ((ss/n1)**0.5)
        temp_mean.append(cbar)
        temp_sd.append(sd)

        ret1 = []
        est1 = []
        S1 = MMGBM1(5, Ht, X_theta, Mu, Sigma) #Stores the stock Prices values using MMGBM fn.
        ret1 = [(S1[i]-S1[i-1])/S1[i-1] for i in range(1,len(S1))]    
        length2 = len(ret1)
        for i3 in range(1,length2-n):
            est1.append((1/dt)*(1/n)*sum(ret1[i3:i3+n]))
        m_est1 = est1
        c_dur1 = dur(m_est1,0)
        cbar1 = np.mean(c_dur1)
        n2 = len(c_dur1)
        ss1 = sum((x-cbar1)**2 for x in c_dur1)
        sd1 = ((ss1/n2)**0.5)
        temp_mean.append(cbar1)
        temp_sd.append(sd1)
    stdv_antithetic.append(sc.pstdev(temp_mean))   
    t_d_mean.append(temp_mean)
    t_d_sd.append(temp_sd)
    
#stdvclassical = [0.31632061422022073,
# 0.42903793378026156,
# 0.5821044176775061,
# 0.7275468194138979,
# 0.7367436476205984,
# 1.0276214719210823,
# 0.6707170817017336,
# 1.1586302636991557,
# 1.2116296883555175,
# 1.4481372367189937,
# 1.495399815905955,
# 1.8082823042529526,
# 1.3107893195333633,
# 1.4402182814927325,
# 1.8342045849595152,
# 1.5228667923144306,
# 2.172908634435843,
# 1.5895028610811124,
# 2.040602537057576]
#------------------------------------------------------------------------------    
df=pd.DataFrame(OrderedDict({'theta%0.1f'%theta_1[i]:t_d_mean[i] for i in range(length)}))
df.boxplot(rot = 90)

print("--- %.3f seconds ---" % (time.clock() - start))