

#            Import Module
#--------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.integrate import solve_ivp, odeint
from random import uniform
import time


# Plan for Demo. Demonstrate two oscilator Kuramoto Model
# Demonstrate multiple Oscillators and plot coherence as a function of time
# Demonstrate three nodes and plot signal next to them
np.random.seed(100)
N = 3
t = (0,1000)
omega = np.array([uniform(1,3) for i in range(N)])
y0 = np.array([uniform(0,2*np.pi) for i in range(N)])

K = 3

# Need to streamlin this portion.... lot of computation
def kuraN(t,theta):
	dthetadt = []
	for i in range(N):
		sums = 0 
		for j in range(N):
			sums += np.sin(theta[j] - theta[i])
		dthetadt += [omega[i] + (K/N)*(sums)]
	return dthetadt	

	
def kuraNpr(t,theta):
	return omega + (K/N)*np.sum(np.sin(y0[:,None]-y0),axis=0)

print(kuraN(t,y0))
print(kuraNpr(t,y0))	

sol1 = solve_ivp(kuraN,t,y0)
sol2 = solve_ivp(kuraNpr,t,y0)

print(kuraN(t,y0))
print(kuraNpr(t,y0))

print(sol1)
print(sol2)





	



	








