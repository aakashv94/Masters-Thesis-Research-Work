# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 10:37:29 2018

@author: Aakash
"""

import numpy as np
import math
import matplotlib.pyplot as plt

'''
Code for finding out the holding time given lambda Vectors in the form
lambda = (lambda1, lambda2) using the exponential distribution.
'''
dt = 1/(250*360) 
def Hold(T, theta):
    c,x,h = 0,0,[]
    while c<=T:
        c_last=c
        h1 = np.random.exponential(scale = theta[x])
        c+=h1
        h.append(min(h1, T-c_last))
        x = 1-x
    return h
#print(list(Hold(1/dt,[0.3,0.4])))
#time_horizon = float(input("Enter time-horizon: \n"))
p = 0.1
p_m = p/(1-p)

#theta_1=[10]
theta_1 = np.arange(10.0,20.5,0.5)
theta_2 =9*theta_1
length = len(theta_1)
#l1= 1/theta_1 
#l2 = p_m*l1
T = 0.6/dt
H_t = [list(Hold(T,[theta_1[i],theta_2[i]])) for i in range(length)]
#H_t = [H_list[i][:-1] for i in range(len(H_list))]
#st_t = [H_list[i][-1] for i in range(len(H_list))]
#defining a function to find states for given holding time vectors

st_t= []
for j1 in range(0,length):
    x, st = 0, [] 
    for j2 in range(0, len(H_t[j1])):
        st.append(x)  
        x = 1-x
    st_t.append(st)
#print(st_t)



mu_bar = (0.9)*0.06 + 0.1*(-0.01)
mu_1 = [-0.01]*length
mu_2 = [0.06]*length
sigma_c = [0.06]*length
sigma = list(zip(sigma_c, sigma_c))
mu = list(zip(mu_1, mu_2))

#---------------------------------------------------------------------------
#Code for Brownian Motion
#---------------------------------------------------------------------------


def generate_normal(mu, variance):
    sigma = np.sqrt(variance)
    U = np.random.uniform(0,1)
    V = np.random.uniform(0,1)
    X1 = (np.sqrt(-2 * np.log(1- U)))*(math.cos(2*(np.pi)*V))
    return (mu + sigma*X1)

#---------- Code for Markov Modulated Geometric Brownian Motion ---------------

def MMGBM(St_Price, Hold_t, state, Mu, Sigma):
    sdt = 1 #in min unit
    S =[St_Price]
    m=0
    a = S[0]
    while m<=len(state)-1:
        H = Hold_t[m]
        for t in np.arange(dt,H,sdt):
            dt_1=min(dt,(H-t)*dt)
            a = a*np.exp((Mu[m] - 0.5*(Sigma[m]**2))*dt_1 + Sigma[m]*(generate_normal(0, abs(dt_1))))
            S.append(a)    
        m+=1
    return S
#
#m = mu[3]
#s = sigma[3]
#Mu = [m[i] for i in st_t[3]]
#Sigma = [s[j] for j in st_t[3]]
#ot = MMGBM(4,H_t[3], st_t[3], Mu, Sigma)
#print(ot,plt.plot(ot), len(ot))
Output = []
for i1 in range(0,length):
    s_t = st_t[i1]
    Ht = H_t[i1]
    m = mu[i1]
    s = sigma[i1]
    Mu = [m[i] for i in s_t]
    Sigma = [s[j] for j in s_t]
    Output.append(MMGBM(5, Ht, s_t, Mu, Sigma))

m_ret = []
est = []
m_est = []
dura = []
d_mean = []
d_sd = []
St_diff = []
n=20 #This is the window size for durations
for i2 in range(0,length):
    ret = []
    est = []
    S = Output[i2]
    ret = [(S[i]-S[i-1])/S[i-1] for i in range(1, len(S))]
    m_ret.append(ret)
#2nd loop to find estimator of drift at jth time.
    r = m_ret[i2]
    for i4 in range(1,len(r)-n):
        est.append((1/dt)*(1/n)*sum(r[i4:i4+n]))
    m_est.append(est)    

r_bar = []
for i3 in range(0, len(m_ret)):
    r_bar.append(np.mean(m_ret[i3])/dt)
    
print("Return mean values are: ", r_bar)
#print(m_est)
def dur(est_list,p): 
    d = []
    flag = 1
    for j1 in range(0,len(est_list)):
        if est_list[j1]*flag<p*flag:
            d.append(j1)
            flag = -flag
    durations = np.diff(d)[::2]
    return durations

for m2 in range(0,length):
    durr=m_est[m2]
#    per=np.percentile(durr,np.arange(0,100,10))[1]
    c_dur = dur(durr,0)
    dura.append(c_dur)
    cbar = np.mean(c_dur)
    n1 = len(c_dur)
    d_mean.append(cbar)
    ss = sum((x-cbar)**2 for x in c_dur)
    d_sd.append((ss/n1)**0.5)

'''
Plotting the values of lambda_1 and lambda_2 vs d_mean and d_sd.
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
##fig2 = plt.figure()
##ax1 = fig2.add_subplot(111)
##ax1.set_xlim(0,12)
##ax1.set_ylim(min(d_mean)-1,max(d_mean)+1)
##plt.title('$\lambda_{2}$ vs mean(duration_drift)')
##plt.xlabel('$\lambda_{2}$')
##plt.ylabel('mean(duration_drift)')
##plt.plot(l2, d_mean, '-b')
#
fig2 = plt.figure()
ax1 = fig2.add_subplot(111)
ax1.set_xlim(min(theta_1)-0.5,max(theta_1)+0.5)
ax1.set_ylim(min(d_sd)-0.5,max(d_sd)+0.5)
plt.title(r'$\theta_{1}$ vs std (squeezed duration)')
plt.xlabel(r'$\theta_{1}$')
plt.ylabel('std(squeezed duration)')
plt.plot(theta_1, d_sd, 'bo')

#fig4 = plt.figure()
#ax1 = fig4.add_subplot(111)
#ax1.set_xlim(0,12)
#ax1.set_ylim(min(d_sd)-50,max(d_sd)+100)
#plt.title('$\lambda_{2}$ vs std(duration_drift)')
#plt.xlabel('$\lambda_{2}$')
#plt.ylabel('std(duration_drift)')
#plt.plot(l2, d_sd, 'b^')

#fig3 = plt.figure()
#ax1 = fig3.add_subplot(111)
#ax1.set_xlim(0,21)
#ax1.set_ylim(min(d_mean)-1,max(d_mean)+1)
#plt.title('$\lambda_{1}$ vs mean (squeezed duration)')
#plt.xlabel('$\lambda_{1}$')
#plt.ylabel('mean(squeezed duration)')
#plt.plot(l1, d_mean, 'go')
#
#fig4 = plt.figure()
#ax1 = fig4.add_subplot(111)
#ax1.set_xlim(0,21)
#ax1.set_ylim(min(d_sd)-0.5,max(d_sd)+0.5)
#plt.title('$\lambda_{1}$ vs std (squeezed duration)')
#plt.xlabel('$\lambda_{1}$')
#plt.ylabel('std(squeezed duration)')
#plt.plot(l1, d_sd, 'yo')

