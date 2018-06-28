# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 11:39:10 2017

@author: Aakash
"""

import numpy as np
import random
import numpy as np

P = np.array([[0,0.67,0.33],[0.5,0,0.5],[0.33,0.67,0]])
P1 = np.zeros((3,3))
#def Cum_Matrix(P,k):
P1[:,0] += P[:,0]
for j in range(1,3):
    P1[:,j] = P1[:,j-1]+ P[:,j]

'''
Function to define the state transition in each step
#Defining Function F with arguement
k-is the number of states
i-current state
x-random numbers from uniform distribution
'''

def F(x,i,k):
    for j in range(0,k):
        if j==0:
            if x <= P1[i-1][0]:
                return 1
        elif 0<j<=k-1:
            if P1[i-1][j-1]<x<=P1[i-1][j]:
                return j+1
        else:
            return k
'''
Markov Chain : Inputs
'''
N = int(input("Enter number of steps: \n"))
states = int(input("Enter number of states: \n"))
start = int(input("Enter intial state: \n"))

def Markov(N,states,start):
    X = [start]
    n = 0
    while n<=N:
        x = X[n]
        U = np.random.uniform(0,1)
        X.append(F(U,x,states))
        n+=1
    return X

print("Markov Chain for simulation with given intial distributionn is : ", Markov(N,states,start))

