# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 18:23:06 2017

@author: Aakash
"""

import random
import numpy as np

#Defining function To Convert Rate Matrix 'Q' To TPM Matrix 'P'

def RM_to_TPM(Q):
    k = len(Q)
    P = np.zeros((k,k))
    for i in range(0,k):
            if Q[i][i] == 0:
                P[i][i] = 1
            if Q[i][i]!=0:
                for j in range(0,k):
                    if j!=i:
                        P[i][j] = Q[i][j]/-Q[i][i]

    return P

#Defining function to find Cumulative Sum Matrix for TPM matrix P
    
def Cum_Sum(Q):
    k = len(Q)
    P1 = np.zeros((k,k))
    #def Cum_Matrix(P,k):
    P1[:,0] = RM_to_TPM(Q)[:,0]
    for j in range(1,k):
        P1[:,j] = P1[:,j-1]+ RM_to_TPM(Q)[:,j]
    return P1


'''
Function to define the state transition in each step
#Defining Function F with arguement
k-is the number of states
i-current state
x-random numbers from uniform distribution
'''

#states = [0,1,2,........,k-1]
'''
Function to define the state transition in each step
'''
def F(Q,x,i):
    k = len(Q)
    for j in range(0,k):
        if x <= Cum_Sum(Q)[i][0]:
            return 0
        if Cum_Sum(Q)[i][j-1]<x<=Cum_Sum(Q)[i][j]:
            return j

def CTMC(Q,start,T):
    '''
    Code to simulate a continuous-time Markov Chain on a finite sample space
    ## Q is kxk matrix as provided, where k = dim(Q)
    ## start is the starting state(a number between 0 an k-1)
    ## T is the total time to run the simulation
    ##
    ##Output is:
    ## ec = embedded chain (sequence of states)
    ## t = time at each state

    '''
    k = len(Q)
    t = [0]
    n=0
    X =  [start]
    # drawing from exponential distribution
    H0 = np.random.exponential(scale = -1/Q[X[n]][X[n]])
    t.append(t[0]+H0)
    while max(t)<T:
        U = np.random.uniform(0,1)
        X.append(F(Q,U,X[n]))
        Hn = np.random.exponential(scale = -1/Q[X[n]][X[n]])
        t.append(max(t) + Hn)
        n+=1
    t.remove(t[0])
    last = t[-1]
    t.remove(t[-1])
    X.remove(X[0])
    print(t,"\n")
    print("Last element of time sequence is: ", last,"\n")
    
    return X


print("\nState Space considered is : [0, 1, 2,.....,k]","\n")
Q_mat = np.array([[-1.0,0.67,0.33],[0.5,-1.0,0.5],[0.33,0.67,-1.0]])
print("Q-matrix entered is : \n", Q_mat,"\n")
print("TPM-matrix for given Q-matrix is: \n", RM_to_TPM(Q_mat))
initial_state = int(input("Enter initial state: "))
time_horizon = int(input("Enter Time-Horizon: "))
print("\n","Time sequence is: \n ")
print("Embeded Chain for given Q-matrix is :\n\n",CTMC(Q_mat,initial_state,time_horizon))

