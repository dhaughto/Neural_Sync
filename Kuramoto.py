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
from scipy.integrate import solve_ivp,odeint
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
	
		return side.real		# Not sure if i can put the abs here though but it
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
#	Two Nodes/ Two Oscilator per node
#---------------------------------------

def two_nodes(C,t):
	print()
	omega = np.random.uniform(0.1,3,size=(1,4))[0]
	# Connection Matrix at random but not needed for example of two
	conn = np.random.choice([0,1],p = [0.0,1.0],size=(2,2))
	np.fill_diagonal(conn,0)
	# Iniitial conditions
	y0 = np.random.uniform(0,np.pi,size=(1,4))[0]
	print('Initial Conditions: ')
	print(y0)
	print()
	print('Connectivity Matrix Randomly Generated:')
	print(conn)
	# Random K values
	print()
	print('K Values Randomly generated:')
	K = np.random.uniform(0.1,6,size=(1,2))[0]

	print(K)

	print()
	print('Global weight C:')
	print(C)
	print()

	# Need to use reduced version

	def kuratwo(t,theta):
		dthetadt = [omega[0] + (K[0]/2)*np.sin(theta[1] - theta[0])+C*(conn[0,1]/2)*(np.sin(theta[2]-theta[0])+np.sin(theta[3]-theta[0])),
					omega[1] + (K[0]/2)*np.sin(theta[0] - theta[1])+C*(conn[0,1]/2)*(np.sin(theta[2]-theta[1])+np.sin(theta[3]-theta[1])),
					omega[2] + (K[1]/2)*np.sin(theta[3] - theta[2])+C*(conn[1,0]/2)*(np.sin(theta[0]-theta[2])+np.sin(theta[1]-theta[2])),
					omega[3] + (K[1]/2)*np.sin(theta[2] - theta[3])+C*(conn[1,0]/2)*(np.sin(theta[0]-theta[3])+np.sin(theta[1]-theta[3]))]
		return dthetadt
		
	sol = solve_ivp(kuratwo,t,y0)
	r = [1,1]

	node_1 = []
	for i in range(len(sol.y[0])):
		node_1 += [[sol.y[0][i],sol.y[1][i]]]
	

	node_2 = []
	for i in range(len(sol.y[0])):
		node_2 += [[sol.y[2][i],sol.y[3][i]]]

# Animation Portion
#------------------------------------------

	fig = plt.figure()
	ax = plt.subplot(211, polar=True)

	point, = ax.plot(node_1[0],r,'go')

	def ani(i):
		point.set_data([node_1[i]],[r])
		return point

	
	ani = animation.FuncAnimation(fig,ani,frames = len(sol.y[0]) ,interval = 500)

	ax = plt.subplot(212,polar=True)

	line, = ax.plot(node_2[0],r,'go')

	def ani1(i):
		line.set_data([node_2[i]],[r])
		return line

	ani1 = animation.FuncAnimation(fig,ani1,frames=len(sol.y[0]),interval=500)	

	plt.show()			

#--------------------------------------------
#---------------------------------------
#  Multiple Nodes/ Multiple Oscillators
#---------------------------------------	

# This is too slow but finish this code then in new sheet can 
# try to make it faster



def multiple_nodes(P,N,C,t):
	# Omega Values
	omega = np.random.uniform(0.1,3,size=(1,P*N))[0]
	# Connectivity Matrix
	conn = np.random.choice([0,1],p = [0.4,0.6],size=(P,P))
	np.fill_diagonal(conn,0)
	# Flatten out Conn Matrix
	new_conn = []
	for i in conn:
		for j in range(N):
			new_conn += [i]
	conn = np.asarray(new_conn)
	# Initial Conditions
	y0 = np.random.uniform(0,np.pi,size=(1,P*N))[0]
	# K values extended for loop
	K = np.random.uniform(0.1,3.5,size=(1,P))[0]
	new_K = []
	for i in K:
		for j in range(N):
			new_K += [i] 
	K = np.asarray(new_K)			

	# Versatile Kuramoto Equation
	def kura_node(t,theta):
		#---------------------------------------------
		dthetadt = []
		for i in range(P*N):
			
			sums_1 = 0
			
			for j in range(P*N):
				sums_1 += np.sin(theta[j] - theta[i])
				
			
			sums_con = 0
			
			for q in range(P-1):
				for m in range(P*N):
					sums_con += (conn[i,q]/N)*np.sin(theta[m] - theta[i])
					
			dthetadt += [omega[i] + ((K[i]/N)*sums_1) + C * sums_con]
		return dthetadt
		#---------------------------------------------

	sol = solve_ivp(kura_node,t,y0)

	r = [1] * N

	x = np.vsplit(sol.y,P)
	
	# Animation Portion of Code
	#---------------------------------------------
	# First Node
	
	fig = plt.figure()
	ax = plt.subplot(321, polar=True)
	scat1 = ax.scatter(x[0][:,0],r)

	def ani1(i):
		scat1.set_offsets(np.c_[x[0][:,i],r])
		return scat1,

	ani1 = animation.FuncAnimation(fig,ani1,frames = len(sol.t),interval = 500)
	#---------------------------------------------
	# Second Node
	ax = plt.subplot(322,polar=True)
	scat2 = ax.scatter(x[1][:,0],r)

	def ani2(i):
		scat2.set_offsets(np.c_[x[1][:,i],r])
		return scat2,

	ani2 = animation.FuncAnimation(fig,ani2,frames = len(sol.t),interval = 500)
	#---------------------------------------------
	#Third Node
	ax = plt.subplot(323,polar=True)
	scat3 = ax.scatter(x[2][:,0],r)

	def ani3(i):
		scat3.set_offsets(np.c_[x[2][:,i],r])
		return scat3,

	ani3 = animation.FuncAnimation(fig,ani3,frames = len(sol.t),interval = 500)
	#---------------------------------------------
	# Fourth Node
	ax = plt.subplot(324,polar=True)
	scat4 = ax.scatter(x[3][:,0],r)

	def ani4(i):
		scat4.set_offsets(np.c_[x[3][:,i],r])
		return scat4,

	ani4 = animation.FuncAnimation(fig,ani4,frames = len(sol.t),interval = 500)
	#---------------------------------------------
	# Fifth Node
	ax = plt.subplot(325,polar=True)
	scat5 = ax.scatter(x[4][:,0],r)

	def ani5(i):
		scat5.set_offsets(np.c_[x[4][:,i],r])
		return scat5,

	ani5 = animation.FuncAnimation(fig,ani5,frames = len(sol.t),interval = 500)
	#---------------------------------------------
	# Sixth Node
	ax = plt.subplot(326,polar=True)
	scat6 = ax.scatter(x[5][:,0],r)

	def ani6(i):
		scat6.set_offsets(np.c_[x[5][:,i],r])
		return scat6,

	ani6 = animation.FuncAnimation(fig,ani6,frames = len(sol.t),interval = 500)

	plt.show()


		
start_time = time.time()	
multiple_nodes(6,5,6,(0,20))
elapsed_time = time.time() - start_time
print(elapsed_time)

