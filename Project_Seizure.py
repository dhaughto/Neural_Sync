


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.integrate import solve_ivp, odeint
from random import uniform
import time



def multiple_nodes(P,N,C,t):
	# Omega Values
	omega = np.random.uniform(0.1,4,size=(1,P*N))[0]
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
	y0 = np.random.uniform(0,2*np.pi,size=(1,P*N))[0]
	# K values extended for loop
	K = np.random.uniform(0.1,2.5,size=(1,P))[0]
	new_K = []
	for i in K:
		for j in range(N):
			new_K += [i] 
	K = np.asarray(new_K)			

	# Versatile Kuramoto Equation
	def kura_node(t,theta):
		return (
			omega + 
			K*np.sum(np.sin(theta.T - theta), axis=1)/N + 
			C*np.einsum('iq,ji->i', conn/N , np.sin(theta.T - theta))
		)
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


# 8 Minutes to produce	
start_time = time.time()	
multiple_nodes(6,3,6,(0,20))
elapsed_time = time.time() - start_time
print(elapsed_time)
