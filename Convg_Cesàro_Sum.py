import numpy as np
import random
import pandas as pd
from numpy import linalg as LA
import matplotlib.pyplot as plt
from decimal import *
P = np.array([[0,0.67,0.33],[0.5,0,0.5],[0.33,0.67,0]])

#def Cum_Matrix(P,k):


def Cum_Sum(P):
    k = len(P)
    P1 = np.zeros((k,k))
    #def Cum_Matrix(P,k):
    P1[:,0] = P[:,0]
    for j in range(1,k):
        P1[:,j] = P1[:,j-1]+ P[:,j]
    return P1

'''
Function to define the state transition in each step
#Defining Function F with arguement
k-is the number of states
i-current state
x-random numbers from uniform distribution
'''
def F(P,x,i):
    k = len(P)
    for j in range(0,k):
        if x <= Cum_Sum(P)[i][0]:
            return 0
        if Cum_Sum(P)[i][j-1]<x<=Cum_Sum(P)[i][j]:
            return j

'''
Defining a function called Markov which takes Number of steps and initial state
as input and returns Markov Chain as output.
Markov Chain :
'''


def Markov(N,start):
    X = [start]
    n = 0
    while n<=N:
        x = X[n]
        U = np.random.uniform(0,1)
        X.append(F(P,U,x))
        n+=1
    return X

'''
Defining function EPM to compute EPM aka Empirical Probability Matrix after
n number of steps using Markov Chain Computed above.
'''
#User Input for the code

start = int(input("Enter intial state: \n"))

steps = int(input("Enter number of steps to simulate: \n"))


X1 = Markov(steps,start)

def EPM(n):
    X = X1[:n]
    #Finding EPM Matrix
    count0_0,count0_1,count0_2 =0,0,0
    count1_0,count1_1,count1_2 =0,0,0
    count2_0,count2_1,count2_2 =0,0,0
    for i in range(1,len(X)):
        if X[i-1]==0:
            count0_0+=1
        if X[i-1]==0 and X[i]==1:
            count0_1+=1
        if X[i-1]==0 and X[i]==2:
            count0_2+=1
        if X[i-1]==1 and X[i]==0:
            count1_0+=1
        if X[i-1]==1:
            count1_1+=1
        if X[i-1]==1 and X[i]==2:
            count1_2+=1
        if X[i-1]==2 and X[i]==0:
            count2_0+=1
        if X[i-1]==2 and X[i]==1:
            count2_1+=1
        if X[i-1]==2:
            count2_2+=1
    p00 = 0
    p11 = 0
    p22 = 0
    p01,p02 = round(count0_1/count0_0,8),round(count0_2/count0_0,8)
    p10,p12 = round(count1_0/count1_1,8),round(count1_2/count1_1,8)
    p20,p21 = round(count2_0/count2_2,8),round(count2_1/count2_2,8)
    X_sim = np.array([[p00,p01,p02],[p10,p11,p12],[p20,p21,p22]])
    return X_sim

'''
Defining a Function "Convg" to find rates of convergence of EPM to TPM matrix
'''

def Convg(TPM):
    alpha = []
    index = []
    term = []
    ter = []
    n = 100
    while n<=10000:
        getcontext().prec = 6
        term1 = LA.norm(EPM(n)-TPM)
        term2 = LA.norm(EPM(2*n)-TPM)
        temp = term1/term2
        term.append(temp)
        ter.append(term1)
        alpha_n = abs(round((np.log2(temp)),8))
        alpha.append(alpha_n)
        index.append(n)
        n = n+300
    y = np.zeros(len(alpha))
    y[0] = alpha[0]
#Defining Cesàro sum on the sequence of rates
    for i in range(1,len(alpha)):
        getcontext().prec = 6
        y[i] = ((i/(i+1))*(y[i-1])) + (alpha[i]/(i+1))

    fig = plt.figure(figsize=(10,8))
    ax1 = fig.add_subplot(111)
    plt.xlabel('n', fontsize=14)
    ax1.scatter(index, ter, marker = 'o', label= r'y: ${\hat{p}_{ij}}^{N} - p_{ij}$')
    ax1.scatter(index, alpha, marker = '^', label= r'$y: \alpha_{N}$')
    ax1.scatter(index, y, marker = 's', label= r'$y: \overline{\alpha_{N}}$')
#    ax1.scatter(index, y,marker = 'o', label='y: Cesàro sum')
    plt.legend(loc='upper right')
    plt.savefig('Comparison_Plot.png')
    return alpha,ter,index


#improve the precision of p^ij - pij
#Ouput
print("Convergence rate to TPM matrix : \n", Convg(P))
