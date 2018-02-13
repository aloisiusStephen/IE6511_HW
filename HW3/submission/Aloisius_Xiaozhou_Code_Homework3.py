
# coding: utf-8

# # IE6511 Homework 3
# Done by: Aloisius Stephen and Yang Xiaozhou

# In[20]:


import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

np.set_printoptions(precision=3)

font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : 12}

matplotlib.rc('font', **font)


# ## Question 1.  Markov Chain Configuration Graph with Random Walk
# a) ![1a.JPG](attachment:1a.JPG) <br>
# 
# 
# b) transition probability matrix
# \begin{aligned}
# \Theta &= 
# \begin{bmatrix}
# \frac{1}{3} & \frac{1}{3} & \frac{1}{3} \\
# \frac{1}{2} & \frac{1}{2} & 0 \\
# \frac{1}{2} & 0 & \frac{1}{2}
# \end{bmatrix}
# \end{aligned}
# <br>
# 
# c) convergence probability matrix
# Let $\pi = [ a \quad b \quad b ]$ where $a,b \in [0,1]$ and $a+b+b=1$, 
# <br>
# 
# 
# \begin{aligned}
# \pi &= \pi \Theta\\
# \begin{bmatrix}
# a & b & b \\
# \end{bmatrix}
# &=
# \begin{bmatrix}
# a & b & b \\
# \end{bmatrix}
# \begin{bmatrix}
# \frac{1}{3} & \frac{1}{3} & \frac{1}{3} \\
# \frac{1}{2} & \frac{1}{2} & 0 \\
# \frac{1}{2} & 0 & \frac{1}{2}
# \end{bmatrix} \\
# &\Rightarrow \frac{1}{3}a + \frac{1}{2}b + \frac{1}{2}b = a\\
# &\Rightarrow b = \frac{2}{3}a \\
# \end{aligned}
# 
# 
# Substituting to $a+b+b=1 \Rightarrow a+\frac{2}{3}a+\frac{2}{3}a = 1 \Rightarrow a = \frac{3}{7}$
# 
# <br>
# Therefore solving for $b = \frac{2}{3}a = \frac{2}{7}$, we get
# \begin{aligned}
# \pi &= 
# \begin{bmatrix}
# \frac{3}{7} & \frac{2}{7} & \frac{2}{7} \\
# \end{bmatrix}
# \end{aligned}
# <br>
# 
# The values of $\pi_2,\pi_3$ are equal because the random walk from S2 and S3 are similar, such that they can only go to either S1 or stay where they are with probability of 0.5. As for $\pi_1$ being slightly larger than the other two convergence probability, it is because from node S2 and S3 they can only go to S1 with probability of 1/2, however from S1 it can go to S2 or S3 with probability of 1/3.

# ## Question 2. Theory-Configuration Graph and transition probability matrix-Greedy 
# 
# a) ![2a.JPG](attachment:2a.JPG)
# <br>
# 
# b)Transition probability matrix
# \begin{aligned}
# \Theta &= 
# \begin{bmatrix}
# 1 & 0 & 0 \\
# \frac{1}{2} & \frac{1}{2} & 0 \\
# \frac{1}{2} & 0 & \frac{1}{2}
# \end{bmatrix}
# \end{aligned}
# <br>
# 
# c)
# Convergence Probability Matrix
# \begin{aligned}
# \pi &= 
# \begin{bmatrix}
# 1& 0 & 0 \\
# \end{bmatrix}
# \end{aligned}
# <br>
# 
# check:
# \begin{aligned}
# \pi \Theta &=
# \begin{bmatrix}
# 1 & 0 & 0 \\
# \end{bmatrix}
# \begin{bmatrix}
# 1 & 0 & 0 \\
# \frac{1}{2} & \frac{1}{2} & 0 \\
# \frac{1}{2} & 0 & \frac{1}{2}
# \end{bmatrix} 
# = 
# \begin{bmatrix}
# 1 & 0 & 0 \\
# \end{bmatrix}\\
# &= \pi \\
# \end{aligned}
# 
# The values of $\pi_2,\pi_3$ are equal to 0 because once the search moves out of S2 and S3 to S1, it will remain in S1 since there are no better solution, which is why $\pi_1 =1$
# <br>

# ## Question 3. Configuration SA Transition and convergence Theory

# a) Cost for each state:

# In[21]:


def cost(x1,x2):
    return 2*np.power((x1-x2),2) + x1

print("State1=(0,1) Costs= %.4f" %cost(0,1))
print("State2=(1,1) Costs= %.4f" %cost(1,1))
print("State3=(0,0) Costs= %.4f" %cost(0,0))
print("State4=(1,0) Costs= %.4f" %cost(1,0))


# Transition probability
# \begin{align}
# P_{12} &=  P_{13} = \frac{1}{2} \\
# P_{21} &= 0.5 e^\frac{-1}{T}, \quad P_{24}= 0.5 e^\frac{-2}{T}\\
# P_{22} &= 1-0.5 e^\frac{-1}{T} -0.5 e^\frac{-2}{T}\\
# P_{31} &= 0.5 e^\frac{-2}{T}, \quad P_{34} = 0.5 e^\frac{-3}{T}\\
# P_{33} &= 1-0.5 e^\frac{-2}{T}-0.5 e^\frac{-3}{T}\\
# P_{42} &= P_{43} = \frac{1}{2} \\
# \end{align}
# 
# 
# There are no self-loops for states 1 and 4 because its neighbours have lower costs in both cases therefore the state will move to either 2 or 3, and not stay at its original state as there is no potential uphill move that can be rejected.

# b)Transition probability matrix <br>
# For $T = 1$,

# In[22]:


def transProb(T):

    p21 = 0.5*np.exp(-1/T)
    p24 = 0.5*np.exp(-2/T)
    p22 = 1-p21-p24

    print("P_21 = %.6f" %p21)
    print("P_24 = %.6f" %p24)
    print("P_22 = %.6f" %p22)

    p31 = 0.5*np.exp(-2/T)
    p34 = 0.5*np.exp(-3/T)
    p33 = 1-p31-p34
    
    print("\nP_31 = %.6f" %p31)
    print("P_34 = %.6f" %p34)
    print("P_33 = %.6f" %p33)
    
    P = np.matrix([[0,0.5,0.5,0],[p21,p22,0,p24],[p31,0,p33,p34],[0,0.5,0.5,0]])
    print("\nP = ")
    print(P)
    return P

P = transProb(1)


# c)

# In[23]:


def matPower(matrix,power):
    sol = matrix

    for i in range(1,power):
        sol = np.matmul(sol,matrix)
    
    return sol

P_5 = matPower(P,5)
P_10 = matPower(P,10)
P_1000 = matPower(P,1000)

print("\nP^5=")
print(P_5)
print("P^10=")
print(P_10)
print("P^1000=")
print(P_1000)


# The probability distribution will be  $\,[0.087 \quad 0.237 \quad 0.644 \quad 0.032]$

# d) for T= 10

# In[24]:


P = transProb(10)
P_5 = matPower(P,5)
P_10 = matPower(P,10)
P_1000 = matPower(P,1000)

print("\nP^5=")
print(P_5)
print("P^10=")
print(P_10)
print("P^1000=")
print(P_1000)


# The probability distribution will be  $\,[0.236 \quad 0.261 \quad 0.289 \quad 0.214]$

# e) for T= 0.2

# In[25]:


P = transProb(0.2)
P_5 = matPower(P,5)
P_10 = matPower(P,10)
P_1000 = matPower(P,1000)

print("\nP^5=")
print(P_5)
print("P^10=")
print(P_10)
print("P^1000=")
print(P_1000)


# The probability distribution cannot be determined since it has not converged at the 1000th iteration

# f) For T= 1 we can see that the probability of being in the global minimum (state 3) is the highest at 0.644, whereas at T = 10, the probability distribution is quite evenly distributed across the 4 states (ranging from 0.214 - 0.289) which means the probability of being at any state and reaching the global minimum (state 3) is almost as likely. <br>
# As for T = 0.2 even though the probability distribution has yet to converge, from the $P^{1000}$ probability distribution we can see that state 3 has the highest probability regardless of where the starting state is with range of 0.812-0.994. <br> 
# It is therefore important to set an appropriate temperature so that the algorithm does not constantly leave the global minimum (when temperature is too high) or get stuck in a local minimum (when the temperature is too low)

# ## Question 4. GA schema 
# 
# a) None of the Schemata are strictly in H, but all of them has an overlap with H in terms of having some common possible genetic sequence. 
# <br>
# 
# Order of Schemata: <br>
# $o(A) = 2$ <br>
# $o(B) = 2$ <br>
# $o(C) = 3$ <br>
# $o(D) = 3$ <br>
# 
# Defining length of Schemata: <br>
# $\delta(A) = 11-1 = 10 $ <br>
# $\delta(B) = 2-1 = 1 $ <br>
# $\delta(C) = 8-6 = 2 $ <br>
# $\delta(D) = 10-3 = 7 $ <br>
# 
# b) Numbers of crossover sites = 10 <br>
# Probability of surviving crossover: <br>
# schema A prob = 0 <br>
# schema B prob = $\frac{9}{10} = 0.9$ <br>
# schema C prob = $\frac{8}{10} = 0.8$ <br>
# schema D prob = $\frac{3}{10} = 0.3$ <br>
# <br>
# 
# c) Probability of surviving mutation at 0.9 mutation probability <br>
# schema A prob = $0.9^2 = 0.81$ <br>
# schema B prob = $0.9^2 = 0.81$ <br>
# schema C prob = $0.9^3 = 0.729$ <br>
# schema D prob = $0.9^3 = 0.729$ <br>
# <br>
# 
# d) Probability of surviving both mutation and crossover <br>
# schema A prob = $0$ <br>
# schema B prob = $0.9 * 0.9^2 = 0.81$ <br>
# schema C prob = $0.8 * 0.9^3 = 0.5832$ <br>
# schema D prob = $0.3 * 0.9^3 = 0.2187$ <br>
# <br>
