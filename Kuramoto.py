#!/usr/bin/python


# Author: Dillon Haughton
# contact: dhaughto@ucsd.edu

#=============================
#         Phys 139
#=============================
'''
	Following Code is for Phys 139 Project

'''
#--------------------------------
#     Import Modules here
#--------------------------------
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.integrate import solve_ivp
from random import uniform


#--------------------------------------------------------------
# Following Section is testing understanding of Kuramoto Model
#--------------------------------------------------------------

#---------------------------------------
#             Two Oscillators          |
#---------------------------------------

omega = [uniform(1,10) for i in range(2)]
K = 3.2

y0 = [0.1,2] 		# In Radians
t_interval = (0,10000)
'''
def kura2ode(t,y):
	dydt = [omega[0] + K*np.sin(y[1]-y[0])/2, omega[1] + K*np.sin(y[0]-y[1])/2]
	return dydt

sol = solve_ivp(kura2ode,t_interval,y0)

r = [1] * len(sol.t)
theta1 = sol.y[0]
theta2 = sol.y[1]

fig = plt.figure()
ax = plt.subplot(111, polar=True)

point, = ax.plot(theta1[0],r[0],'go')
line, = ax.plot(theta2[0],r[0],'go')

def ani(coords):
	point.set_data([coords[0]],[coords[1]])
	line.set_data([coords[2]],[coords[3]])
	return point, line

def frames():
	for k,j,x,z in zip(theta1,r,theta2,r):
		yield k,j,x,z	

	
ani = animation.FuncAnimation(fig,ani,frames = frames,interval = 500)

plt.show()
'''
#--------------------------------------------------
#                 N number of Oscillators
#--------------------------------------------------
# Number of Oscilators
N = 100
omega = np.array([uniform(1,10) for i in range(N)])
y0 = np.array([uniform(1,100) for i in range(N)])
t = (0,1000)


def kuraN(t,theta):
	dthetadt = []
	
	'''
	dtheta[i] = w[i] + K/N * (sum(sin(theta[j]-theta[i])))
	'''
	for i in range(N):
		sums = 0 
		for j in range(N):
			sums += np.sin(theta[j] - theta[i])

		dthetadt += [omega[i] + (K/N)*(sums)]

	return dthetadt	

sol = solve_ivp(kuraN,t,y0)

r = [1] * len(sol.t)
theta1 = sol.y[0]
theta2 = sol.y[1]

fig = plt.figure()
ax = plt.subplot(111, polar=True)

point, = ax.plot(theta1[0],r[0],'go')
line, = ax.plot(theta2[0],r[0],'go')

def ani(coords):
	point.set_data([coords[0]],[coords[1]])
	line.set_data([coords[2]],[coords[3]])
	return point, line

def frames():
	for k,j,x,z in zip(theta1,r,theta2,r):
		yield k,j,x,z	

	
ani = animation.FuncAnimation(fig,ani,frames = frames,interval = 500)

plt.show()







