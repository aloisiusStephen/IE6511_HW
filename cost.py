import numpy as np
import matplotlib.pyplot as plt

def cost(s):
    return (400 - pow(s-21,2)) * np.sin(s*np.pi/6)

s = np.linspace(0,500,501,True)
plt.plot(s,cost(s))