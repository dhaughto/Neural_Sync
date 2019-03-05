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
import time


#--------------------------------------------------------------
# Following Section is testing understanding of Kuramoto Model
#--------------------------------------------------------------
def two_oscillator():
#---------------------------------------
#             Two Oscillators          |
#---------------------------------------

	omega = [uniform(1,10) for i in range(2)]
	K = 3.2

	y0 = [0.1,2] 		# In Radians
	t_interval = (0,10000)

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

#--------------------------------------------------
#                 N number of Oscillators
#--------------------------------------------------
def multiple_oscillator(N,K,t):
# Number of Oscillators
# Speed things up by adding np arrays wherever you can
# time increased drastically at N = 1000 need to fix

#----------------------------
#     Table of Observation
# K = 6.8 Sync then out of sync
# omenga (1,5)
# y0 (0,2pi)

	omega = np.array([uniform(1,3) for i in range(N)],dtype=np.float16)
	y0 = np.array([uniform(0,np.pi) for i in range(N)],dtype=np.float16)
	

# Need to streamlin this portion.... lot of computation
	def kuraN(t,theta):
		dthetadt = []
	
		for i in range(N):
			sums = 0 
			for j in range(N):
				sums += np.sin(theta[j] - theta[i])
			dthetadt += [omega[i] + (K/N)*(sums)]

		return dthetadt	

	sol = solve_ivp(kuraN,t,y0)

# You want list of each time slice
# add to list sol.y[i][0] next list sol.y[i][1]
# Need to work the data here for animation
	form_at = []
	for i in range(len(sol.y[0])):
		form_half = []	
		form_half += [sol.y[j][i] for j in range(N)]
		form_at += [form_half]


	r = [1] * N
	rl = [r] * len(sol.t)

	fig = plt.figure()
	ax = plt.subplot(121, polar=True)

#---------------------------------------
# Want a time label so you can see what time there is sync
# Also need animation of coherence as well
	point, = ax.plot(form_at[0],rl[0],'go')

	def ani(i):
		point.set_data([form_at[i]],[rl[i]])
	
		return point
	
	ani = animation.FuncAnimation(fig,ani,frames = len(sol.t),interval = 200,repeat=False)


#---------------------------------------
#				Coherence
#---------------------------------------

#? Email Professor ask about coherence 
	def r(theta):
	
		N = len(theta)
		average = np.mean(theta)
	
	
		sums = []
		for i in range(len(theta)):
			sums += [np.exp(np.complex(0,theta[i]))]
			sums = sum(sums)
	
		side = (1/N)*(sums/(np.exp(complex(0,average))))
	
		return abs(side.real)		# Not sure if i can put the abs here though but it
									# seems to be an accurate description

	coupling = [r(i) for i in form_at]

	ax1 = plt.subplot(122)

	line, = ax1.plot(sol.t[0],coupling[0],'-o',markersize=2)

	def ani2(i):
		line.set_data(sol.t[:i],coupling[:i])
		line.axes.axis([0,max(sol.t),0,1])

		return line

	ani2 = animation.FuncAnimation(fig,ani2,frames = len(sol.t),interval=200,repeat=False)	

	plt.show()

# Want a time label so you can see what time there is sync
# Also need animation of coherence as well
# Consider adding connectivity matrix to extend to Neural Networks
# Save as csv for higher order N values which can be animated later. Honestly
# Might run all day
# Need stop watch to count how long it took for different N values
#---------------------------------------

# A seizure can look like a grid of these animations syncing up together
'''
start_time = time.time()
multiple_oscillator(100,3.5,(0,1000))
elapsed_time = time.time() - start_time
print(elapsed_time)
'''
# 
#---------------------------------------
#			Multiple Nodes
#---------------------------------------

def multiple_nodes(N,P,C,t):
	print()
	# N = number of oscillators
	# P = number of nodes
	# K = coupling constant
	# C = weighted global constant
	# t (tuple of time)
	#----------------------------
	# Actually Modeling Epilepsy
	#----------------------------
	# 6 is an abritrary number of nodes P
	# p goes from 1 - P
	# i goes from 1 - N
	# q goes from 1 - P

	# Omega is a matrix N columns, P rows
	# K list P long
	# C is an integer
	
	omega = np.random.uniform(0.1,3,size=(P,N))
	# Connection Matrix at random
	conn = np.random.choice([0,1],p = [0.3,0.7],size=(P,P))
	print('Connectivity Matrix Randomly Generated:')
	print(conn)
	# Random K values
	print()
	print('K Values Randomly generated:')
	K = np.random.uniform(0.1,6,size=(P,1))
	np.fill_diagonal(K,0)
	print(K)
	print()
	print('Global weight C:')
	print(C)

	def kurakura(t,theta):
		# initialize matrix
		dthetadt = np.asarray([])
		for p in range(P):
			part_1 = []
			for i in range(N):


				for j in range(N):
					part_1 += [np.sin(theta[p,j]-theta[p,i])]


					part_4 = []
					for q in range(P):
						part_4 += (conn[p,q]/N)*np.sin(theta[q,j]-theta[p,i])
						



thets = omega[p,i]+(K[p,0]/N)*np.sin(theta[p,j]-theta[p,i]) + C * (conn[p,q]/N)*np.sin(theta[q,j]-theta[p,i])


		







		
		
					


multiple_nodes(100,6,2.0,(0,20))	
	


