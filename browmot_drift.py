# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 11:02:47 2017

@author: Aakash
"""

'''
Code for Brownian Motion with Drift
'''
import numpy as np
import matplotlib.pyplot as plt
import math



def generate_normal(mu, variance):
    sigma = np.sqrt(variance)
    U = np.random.uniform(0,1)
    V = np.random.uniform(0,1)
    X1 = (np.sqrt(-2 * np.log(1- U)))*(math.cos(2*(np.pi)*V))
    X2 = (np.sqrt(-2 * np.log(1- U)))*(math.sin(2*(np.pi)*V))
    return (mu + sigma*X1)


def brn_motion(t):
    h = 0.01
    k = int(t/h)
    T = np.zeros(k)
    B_t = np.zeros(k)
    Z = np.zeros(k)
    i=1
    while i<k:
        T[i] = h + T[i-1]
        Z[i] = generate_normal(0,1)
        B_t[i] = B_t[i-1] + (np.sqrt(h))*Z[i]
        i+=1
    return T, B_t

#-------Verifying the code for Brownian motion by plotting histograms--------#
    
#    B_diff = []
#    n = []
#    for i in range(1, len(B_t)):
#        n.append(i)
#        B_diff.append(B_t[i] - B_t[i-1])
#    plt.hist(B_diff, bins=25)
#    plt.xlabel("Numbers")
#    plt.ylabel("Value")

#Output:
#print(brn_motion(5))

#---------------------End of Code for Brownian Motion ------------------#

#---------------Code for brownian motion with drift mu and variance term sigma^2.---------#
    

#
def brn_motion_drift(mu, variance, t):
    sigma = np.sqrt(variance)
    h=0.01
    k = int(t/h)
    X_t = np.zeros(k)
    B_t = brn_motion(t)[1]
#    T = brn_motion(t)[0]
    X_t = sigma*B_t + (mu - 0.5*(sigma**2))*h
    return X_t
    
#    X_diff = []
#    n = []
#    for i in range(1, len(X_t)):
#        n.append(i)
#        X_diff.append(X_t[i] - X_t[i-1])
#    plt.hist(X_diff, bins=25)
#    plt.xlabel("Numbers")
#    plt.ylabel("Value")
#-----------------------------------------------
#Output:
#print(brn_motion_drift(0,0.1,5))
 #----------------------------------------------   

#def brn_motion_drift(mu, variance, t):
#    h = 0.01
#    k = int(t/h)
#    sigma = np.sqrt(variance)
#    T = np.zeros(k)
#    X_t = np.zeros(k)
#    X_t[0]= 0
#    Z = np.zeros(k)
#    Z[0] = generate_normal(0,1)
#    i=1
#    while i<k:
#        T[i] = h + T[i-1]
#        Z[i] = generate_normal(0,1)
#        X_t[i] = X_t[i-1] + sigma*(np.sqrt(h))*Z[i] + mu*(h)
#        i+=1
#    X_diff = []
#    n = []
#    for i in range(1, len(X_t)):
#        n.append(i)
#        X_diff.append(X_t[i] - X_t[i-1])
#    plt.hist(X_diff, bins=25)
#    plt.xlabel("Numbers")
#    plt.ylabel("Value")

'''
To simulate Geometric Brownian motion with drift mu and variance sigma^2.
'''
#
#gbm_d=brn_motion_drift(0.1, 0.01, 5)
#s_t=[20]
#ss=s_t[0]
#for i in range(len(gbm_d)-1):
#    ss1=ss*np.exp(gbm_d[i+1]-gbm_d[i])
#    s_t.append(ss1)
#    ss=ss1
#    
#def gbm(S0, mu, variance, t):
#    h=0.01
#    k = int(t/h)
#    T = brn_motion(t)[0]
#    S_t = np.zeros(k)
#    S_t[0] = S0
#    i=1
#    while i<k:
#        m1 = brn_motion_drift(mu, variance, t)[i]
#        m2 = brn_motion_drift(mu, variance, t)[i-1]
#        S_t[i] = S_t[i-1]*np.exp(m1-m2)
#        i+=1
#    fig = plt.figure()
#    ax1 = fig.add_subplot(111)
#    ax1.set_xlim(0,t+0.25)
#    ax1.set_ylim(min(S_t)-0.25, max(S_t)+0.25)
#    plt.title('GBM: $S_{t}$ vs t')
#    plt.xlabel('time $(t)$')
#    plt.ylabel('$S_{t}$')
#    plt.plot(T, S_t)
###---------------------------------------Check
##ax1.scatter(T, S_t, marker = 'o', color ='yellow')
##ax1.scatter(T, check_St, marker = '*', color = 'black')
##ax1.scatter(T, brn_motion_drift(mu, variance, t), marker = '^', color = 'orange')    
#print(gbm(20, 0.1, 0.01 , 5))


#S0 = 20
#mu=0.01
#variance=0.0001
#h=0.01
#def gbm(t):
#    S0 = 20
#    mu=0.01
#    variance=0.0001
#    h=0.01
#    sigma = np.sqrt(variance)
#    k= int(t/h)
#    Y = np.zeros(k)
#    Z = np.zeros(k)
#    T = np.zeros(k)
#    S_t = np.zeros(k)
#    S_t[0] = S0
#    i=1
#    while i<k:
#        T[i] = T[i-1] + h
#        Z[i] = generate_normal(0,1)
#        Y[i] = np.exp((sigma*np.sqrt(h)*Z[i])+(mu-0.5*sigma**2)*h)
#        S_t[i] = S_t[i-1]*Y[i]
#        i+=1
#    return(S_t)
#S_t=gbm(5)
#plt.plot(S_t, color = 'red')
#    