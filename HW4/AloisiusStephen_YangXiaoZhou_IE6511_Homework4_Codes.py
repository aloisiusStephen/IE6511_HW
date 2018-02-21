import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

np.set_printoptions(precision=3)
np.random.seed(6511)

def bump(x):

    dimen=np.size(x);

    sum1 = 0
    sum2 = 0
    tim1 = 1
    tim2 = 1
    z = np.zeros(dimen)
    
    for i in range(0,dimen):
        sum1=sum1+np.power(np.cos(x[i]),4)
        sum2=sum2+i*np.power(x[i],2)
        tim1=tim1*np.power(np.cos(x[i]),2)
        tim2=tim2*(x[i])
    for i in range(0,dimen):
        if(0<=x[i] and x[i]<=10):
            z[i]=1
        else:
            z[i]=0
            
    if(any(z) and tim2>=0.75):
        y = np.abs((sum1-2*tim1)/(np.sqrt(sum2)))
    else:
        y=0
    
    return y

def DDS(x_min,x_max,m,x_initial):
    r=0.2
    sBest=x_initial
    sCur=x_initial
    CostBest=bump(sBest)
    dimen=np.size(x_initial)
    Cost_iter = np.zeros(m)
    Iter = np.zeros(m)
    x_range=x_max-x_min
    k=0
    
    for i in range(0,m):
        sCur=sBest
        Iter[i] = np.int(i+1)
        Cost_iter[i]=CostBest
        for j in range(0,dimen):
            if (np.random.rand(1)<(1-(np.log(i+1)/np.log(m+1)))):
                k=k+1
                sCur[j]=sBest[j]+np.random.randn(1,1)*r*(x_range)
                if(sCur[j]<x_min):
                    sCur[j]=x_min+(x_min-sCur[j])
                    if(sCur[j]>x_max):
                        sCur[j]=x_min
                        
                if(sCur[j]>x_max):
                    sCur[j]=x_max-(sCur[j]-x_max)
                    if(sCur[j]<x_min):
                        sCur[j]=x_max
        
            if(k==0):
                index=np.random.randint(0,dimen)
                sCur[index]=sBest[index]+np.random.randn(1,1)*r*(x_range)
                if(sCur[index]<x_min):
                    sCur[index]=x_min+(x_min-sCur[index])
                    if(sCur[index]>x_max):
                        sCur[index]=x_min
                            
                if(sCur[index]>x_max):
                    sCur[index]=x_max-(sCur[index]-x_max)
                    if(sCur[index]<x_min):
                        sCur[index]=x_max
            k=0
            if(bump(sCur)>CostBest):
                sBest=sCur
                CostBest=bump(sBest)
    sol = pd.DataFrame(np.column_stack((Iter,Cost_iter)), columns = ['Iteration','Cost'])
    return sol 


x_init = 10*np.random.rand(20)
trial = 20
itr = 500
Trial_best = DDS(0,10,itr,x_init)

for i  in range(1,trial):
    Trial_best = Trial_best.append(DDS(0,10,itr,x_init)) 
    
#average of Best_Cost plot
plt.figure(figsize=[10,8])
plt.plot(Trial_best.groupby('Iteration').mean().Cost, '--')
plt.xlabel('Iteration/Function evaluation')
plt.ylabel('Cost')
plt.legend(['Best_Cost'])
plt.tight_layout()
