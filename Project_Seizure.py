

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
def two_oscillator(K,y0_range,omega_range,t):
	# Give two initial coniditons in list
	# Give time in tupal
	# Give list of two omega values
#---------------------------------------
#             Two Oscillators          |
#---------------------------------------
	omega_range_love = omega_range[0]
	omega_range_high = omega_range[1]
	omega = [uniform(omega_range_love,omega_range_high) for i in range(2)]
	y0 = [uniform(y0_range[0],y0_range[1]) for i in range(2)]

	def kura2ode(t,y):
		dydt = [omega[0] + K*np.sin(y[1]-y[0])/2, omega[1] + K*np.sin(y[0]-y[1])/2]
		return dydt

	sol = solve_ivp(kura2ode,t,y0)

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

	
	ani = animation.FuncAnimation(fig,ani,frames = frames,interval = 800,repeat=False)
	plt.title('Two Synced Oscillators: K value of {}'.format(K))
	plt.show()



#two_oscillator(2.2,[0,np.pi],[0,0.5],(0,30))
#===========================================================================================

def multiple_oscillator(N,K,y0_range,omega_range,t):
# Number of Oscillators
# Speed things up by adding np arrays wherever you can
# time increased drastically at N = 1000 need to fix

#----------------------------
#     Table of Observation
# K = 6.8 Sync then out of sync
# omenga (1,5)
# y0 (0,2pi)

	omega = np.array([uniform(omega_range[0],omega_range[1]) for i in range(N)],dtype=np.float16)
	print(omega)
	y0 = np.array([uniform(y0_range[0],y0_range[1]) for i in range(N)],dtype=np.float16)
	print(y0)
	

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
	plt.title('{} Oscillators'.format(N))

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
		sums = np.sum(sums)
	
		side = (1/N)*(sums/(np.exp(complex(0,average))))
	
		return abs(side)		# Not sure if i can put the abs here though but it
									# seems to be an accurate description
	coupling = []								
	for i in form_at:
		coupling += [r(i)]
									

	ax1 = plt.subplot(122)

	line, = ax1.plot(sol.t[0],coupling[0],'-o',markersize=2)

	def ani2(i):
		line.set_data(sol.t[:i],coupling[:i])
		line.axes.axis([0,max(sol.t),0,1])

		return line

	ani2 = animation.FuncAnimation(fig,ani2,frames = len(sol.t),interval=200,repeat=False)	

	plt.title('Coherence K Value: {}'.format(K))
	plt.show()	

#multiple_oscillator(100,4.8,[0,2*np.pi],[0.1,3.1],(0,50))
#multiple_oscillator(100,5.5,[0,2*np.pi],[0.1,6.1],(0,100))
# Want Random Coherence 
# Want For Sure Coherenence
# Want Coherence and steep drop off


#multiple_oscillator(100,5.5,[0,2*np.pi],[0.1,6.1],(0,100))
#===========================================================================================

def multiple_nodes(N,C,omega_range,K_range,y0_range,t):
	P = 3
	start_time = time.time()
	# Omega Values
	omega = np.random.uniform(omega_range[0],omega_range[1],size=(1,P*N))[0]
	# Connectivity Matrix
	conn = np.random.choice([0,1],p=[0.4,0.6],size=(P,P))
	np.fill_diagonal(conn,0)
	print("Connectivity Matrix:")
	print(conn)
	print()

	new_conn = []
	for i in conn:
		for j in range(N):
			new_conn += [i]
	conn = np.asarray(new_conn)
	#Initial Conditions
	y0 = np.random.uniform(y0_range[0],y0_range[1],size=(1,P*N))[0]
	# K Values
	K = np.random.uniform(K_range[0],K_range[1],size=(1,P))[0]
	print('K Values')
	print(K)
	print()
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

	#------------------------------
	#     Calculate Coherenece
	#------------------------------
	def coherence(theta):
		N = len(theta)
		average = np.mean(theta)

		sums = []
		for i in range(len(theta)):
			sums+=[np.exp(np.complex(0,theta[i]))]
		sums = np.sum(sums)
		
		side = (1/N)*(sums/(np.exp(complex(0,average))))

		return abs(side)

		
	coupling1 = []
	coupling2 = []
	coupling3 = []
	for i in range(len(sol.t)):
		coupling1 += [coherence(x[0][:,i])]
		coupling2 += [coherence(x[1][:,i])]
		coupling3 += [coherence(x[2][:,i])]

	
	
	#--------------------------------------------------
	#                Animation Section
	#--------------------------------------------------
	# Animation 1
	fig = plt.figure()
	ax = plt.subplot(321,polar=True)
	scat1 = ax.scatter(x[0][:,0],r)

	def ani1(i):
		scat1.set_offsets(np.c_[x[0][:,i],r])
		return scat1,

	ani1 = animation.FuncAnimation(fig,ani1,frames=len(sol.t),interval = 100, repeat=False)

	plt.title('Nodes')
	# Animation 2

	ax = plt.subplot(322)
	sign, = ax.plot(sol.t[0],coupling1[0],'-o',markersize=2)

	def ani2(i):
		sign.set_data(sol.t[:i],coupling1[:i])
		sign.axes.axis([0,max(sol.t),0,1])
		return sign

	ani2 = animation.FuncAnimation(fig,ani2,frames=len(coupling1),interval = 100, repeat=False)

	plt.title('Generalized EEG')
	# Animation 3
	ax = plt.subplot(323,polar=True)
	scat2 = ax.scatter(x[1][:,0],r)

	def ani3(i):
		scat2.set_offsets(np.c_[x[1][:,i],r])
		return scat2,

	ani3 = animation.FuncAnimation(fig,ani3,frames=len(sol.t),interval = 100, repeat=False)	

	# Animation 4
	ax = plt.subplot(324)
	sign2, = ax.plot(sol.t[0],coupling2[0],'-o',markersize=2)

	def ani4(i):
		sign2.set_data([sol.t[:i]],[coupling2[:i]])
		sign2.axes.axis([0,max(sol.t),0,1])
		return sign2

	ani4 = animation.FuncAnimation(fig,ani4,frames=len(coupling2),interval=100,repeat=False)	


	# Animation 5
	ax = plt.subplot(325,polar=True)
	scat3 = ax.scatter(x[2][:,0],r)

	def ani5(i):
		scat3.set_offsets(np.c_[x[2][:,i],r])
		return scat3,

	ani5 = animation.FuncAnimation(fig,ani5,frames=len(sol.t),interval=100,repeat=False)

	# Animation 6
	ax = plt.subplot(326)
	sign3, = ax.plot(sol.t[0],coupling3[0],'-o',markersize=2)

	def ani6(i):
		sign3.set_data([sol.t[:i]],[coupling3[:i]])
		sign3.axes.axis([0,max(sol.t),0,1])
		return sign3

	ani6 = animation.FuncAnimation(fig,ani6,frames=len(coupling3),interval=100,repeat=False)	
	
	elapsed_time = time.time() - start_time
	print('Time: ',elapsed_time)
	plt.show()	

#multiple_nodes(20,1.2,[1,3],[1,3],[0,2*np.pi],(0,10))		 	

#===========================================================================================
def EEG1(N,C,omega_range,K_range,y0_range,t):
	P = 3
	start_time = time.time()
	# Omega Values
	omega = np.random.uniform(omega_range[0],omega_range[1],size=(1,P*N))[0]
	# Connectivity Matrix
	conn = np.random.choice([0,1],p=[0.4,0.6],size=(P,P))
	np.fill_diagonal(conn,0)
	print("Connectivity Matrix:")
	print(conn)
	print()

	new_conn = []
	for i in conn:
		for j in range(N):
			new_conn += [i]
	conn = np.asarray(new_conn)
	#Initial Conditions
	y0 = np.random.uniform(y0_range[0],y0_range[1],size=(1,P*N))[0]
	# K Values
	K = np.random.uniform(K_range[0],K_range[1],size=(1,P))[0]
	print('K Values')
	print(K)
	print()
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

	#------------------------------
	#     Calculate Coherenece
	#------------------------------
	def coherence(theta):
		N = len(theta)
		average = np.mean(theta)

		sums = []
		for i in range(len(theta)):
			sums+=[np.exp(np.complex(0,theta[i]))]
		sums = np.sum(sums)
		
		side = (1/N)*(sums/(np.exp(complex(0,average))))

		return abs(side)

		
	coupling1 = []
	coupling2 = []
	coupling3 = []
	for i in range(len(sol.t)):
		coupling1 += [coherence(x[0][:,i])]
		coupling2 += [coherence(x[1][:,i])]
		coupling3 += [coherence(x[2][:,i])]

	
	
	#--------------------------------------------------
	#                Animation Section
	#--------------------------------------------------
	# Animation 1
	fig = plt.figure()
	ax = plt.subplot(321,polar=True)
	scat1 = ax.scatter(x[0][:,0],r)

	def ani1(i):
		scat1.set_offsets(np.c_[x[0][:,i],r])
		return scat1,

	ani1 = animation.FuncAnimation(fig,ani1,frames=len(sol.t),interval = 100, repeat=False)

	plt.title('Nodes')
	# Animation 2
	ax = plt.subplot(322)

	ax.set_ylim(-1,1)
	sign, = ax.plot(sol.t,coupling1[0]*np.sin(sol.t - np.mean(x[0][:,0])))

	def ani2(i):
		sign.set_data([sol.t],[coupling1[i]*np.sin(sol.t - np.mean(x[0][:,i]))])
		return sign,

	ani2 = animation.FuncAnimation(fig,ani2,frames=len(coupling1),interval = 100, repeat=False)

	plt.title('Generalized EEG')
	# Animation 3
	ax = plt.subplot(323,polar=True)
	scat2 = ax.scatter(x[1][:,0],r)

	def ani3(i):
		scat2.set_offsets(np.c_[x[1][:,i],r])
		return scat2,

	ani3 = animation.FuncAnimation(fig,ani3,frames=len(sol.t),interval = 100, repeat=False)	

	# Animation 4
	ax = plt.subplot(324)
	ax.set_ylim(-1,1)
	sign2, = ax.plot(sol.t,coupling2[0]*np.sin(sol.t - np.mean(x[1][:,0])))

	def ani4(i):
		sign2.set_data([sol.t],[coupling2[i]*np.sin(sol.t - np.mean(x[1][:,i]))])
		return sign2,

	ani4 = animation.FuncAnimation(fig,ani4,frames=len(coupling2),interval=100,repeat=False)	


	# Animation 5
	ax = plt.subplot(325,polar=True)
	scat3 = ax.scatter(x[2][:,0],r)

	def ani5(i):
		scat3.set_offsets(np.c_[x[2][:,i],r])
		return scat3,

	ani5 = animation.FuncAnimation(fig,ani5,frames=len(sol.t),interval=100,repeat=False)

	# Animation 6
	ax = plt.subplot(326)
	ax.set_ylim(-1,1)
	sign3, = ax.plot(sol.t,coupling3[0]*np.sin(sol.t - np.mean(x[2][:,0])))

	def ani6(i):
		sign3.set_data([sol.t],[coupling3[i]*np.sin(sol.t - np.mean(x[2][:,i]))])
		return sign3,

	ani6 = animation.FuncAnimation(fig,ani6,frames=len(coupling3),interval=100,repeat=False)	
	
	elapsed_time = time.time() - start_time
	print('Time: ',elapsed_time)
	plt.show()	

# 3 Nodes, 10 each about 6~8 seconds
# 3 Nodes, 20 each about 30 seconds
# 3 Nodes, 100 each about 300 seconds (5 minutes)
# 3 Nodes , 100 each for 30 seconds about 1000 seconds (15 minutes)

# EEG can be used to display this data and how it can resemble an EEG.

#multiple_nodes(3,20,3.3,[3.0,5.5],[0.2,3.6],[0,2*np.pi],(0,30))

EEG1(20,0.6,[1,3],[1,3],[0,2*np.pi],(0,20))

#===========================================================================================
def six_nodes(N,C,K_range,omega_range,y0_range,t,conn=None):
	print()
	P = 6 # number of nodes
	# Omega Values
	omega = np.random.uniform(omega_range[0],omega_range[1],size=(1,P*N))[0]
	# Connectivity Matrix

	if conn == None:
		conn = np.random.choice([0,1],p = [0.4,0.6],size=(P,P))
		np.fill_diagonal(conn,0)
		print('Connectivity Matrix: ')
		print(conn)
	# Flatten out Conn Matrix
		new_conn = []
		for i in conn:
			for j in range(N):
				new_conn += [i]
		conn = np.asarray(new_conn)
	else:
		print('Connectivity Matrix: ')
		print(conn)
	
	# Initial Conditions
	y0 = np.random.uniform(y0_range[0],y0_range[1],size=(1,P*N))[0]
	# K values extended for loop
	K = np.random.uniform(K_range[0],K_range[1],size=(1,P))[0]
	print()
	print('K Values: ')
	print(K)
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
	ax.set_ylim(0,1.5)
	scat1 = ax.scatter(x[0][:,0],r)

	def ani1(i):
		scat1.set_offsets(np.c_[x[0][:,i],r])
		return scat1,

	ani1 = animation.FuncAnimation(fig,ani1,frames = len(sol.t),interval = 100)
	#---------------------------------------------
	# Second Node
	ax = plt.subplot(322,polar=True)
	ax.set_ylim(0,1.5)
	
	scat2 = ax.scatter(x[1][:,0],r)

	def ani2(i):
		scat2.set_offsets(np.c_[x[1][:,i],r])
		return scat2,

	ani2 = animation.FuncAnimation(fig,ani2,frames = len(sol.t),interval = 100)
	#---------------------------------------------
	#Third Node
	ax = plt.subplot(323,polar=True)
	ax.set_ylim(0,1.5)
	scat3 = ax.scatter(x[2][:,0],r)

	def ani3(i):
		scat3.set_offsets(np.c_[x[2][:,i],r])
		return scat3,

	ani3 = animation.FuncAnimation(fig,ani3,frames = len(sol.t),interval = 100)
	#---------------------------------------------
	# Fourth Node
	ax = plt.subplot(324,polar=True)
	ax.set_ylim(0,1.5)
	scat4 = ax.scatter(x[3][:,0],r)

	def ani4(i):
		scat4.set_offsets(np.c_[x[3][:,i],r])
		return scat4,

	ani4 = animation.FuncAnimation(fig,ani4,frames = len(sol.t),interval = 100)
	#---------------------------------------------
	# Fifth Node
	ax = plt.subplot(325,polar=True)
	ax.set_ylim(0,1.5)
	scat5 = ax.scatter(x[4][:,0],r)

	def ani5(i):
		scat5.set_offsets(np.c_[x[4][:,i],r])
		return scat5,

	ani5 = animation.FuncAnimation(fig,ani5,frames = len(sol.t),interval = 100)
	#---------------------------------------------
	# Sixth Node
	ax = plt.subplot(326,polar=True)
	ax.set_ylim(0,1.5)
	scat6 = ax.scatter(x[5][:,0],r)

	def ani6(i):
		scat6.set_offsets(np.c_[x[5][:,i],r])
		return scat6,

	ani6 = animation.FuncAnimation(fig,ani6,frames = len(sol.t),interval = 100)

	plt.show()

# More connections can cause less coherence 
#six_nodes(10,0.6,[1,3.5],[1,8],[0,2*np.pi],(0,10))
