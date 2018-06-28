# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 10:39:18 2017

@author: Aakash
"""

'''
Code for Brownian Motion with Drift
'''

def generate_normal(mu, variance):
    sigma = np.sqrt(variance)
    U = np.random.uniform(0,1)
    V = np.random.uniform(0,1)
    X1 = (np.sqrt(-2 * np.log(1- U)))*(math.cos(2*(np.pi)*V))
    X2 = (np.sqrt(-2 * np.log(1- U)))*(math.sin(2*(np.pi)*V))
    return (mu + sigma*X1)


def brn_motion(t):
    h = 0.01
    T = np.zeros(int(1+t/h))
    B_t = np.zeros(int(1+t/h))
    Z = np.zeros(int(1+t/h))
    Z[0] = generate_normal(0,1)
    i=1
    while T[i-1]+h<=t:
        T[i] = h + T[i-1]
        Z[i] = generate_normal(0,1)
        B_t[i] = B_t[i-1] + (np.sqrt(T[i] - T[i-1])*Z[i])
        i+=1
    return T, B_t

#---------------Code for brownian motion with drift mu and variance term sigma^2.---------#
    
''''
To simulate brownian motion with drift mu and variance sigma^2 , we use 
previously generated brownian motion function("brn_motion") and then use the
relation X(t) = sigma*B(t) + mu*T, where X(t) is a brownian motion with drift.
''''

def brn_motion_drift(mu, variance, t):
    sigma = np.sqrt(variance)
    X_t = np.zeros(int(1+t/h))
    B_T = brn_motion(t)[1]
    T = brn_motion(t)[0]
    X_t = sigma*B_T + mu*T
    plt.plot(T, X_t)

print(brn_motion_drift(0.01, 0.1 , 5))

    