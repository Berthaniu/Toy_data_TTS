import numpy as np 
import matplotlib.pyplot as plt
import math
import pdb
import scipy.special



def PoissonGumbelMax(lam,N):
	# sample uniform 
	c  = np.arange(N, dtype=np.int64)

	# calculate poisson prob.
	x = np.exp(-lam)*np.power(np.int64(lam),np.int64(c))/scipy.special.factorial(c)
	# pdb.set_trace()
	# Generate Gumbel(0)
	yu = np.random.uniform(0,1,N)
	y = -np.log(-np.log(yu+1e-20)+1e-20)
	z = np.log(x+1e-20)+y
	pdb.set_trace()
	return np.argmax(z)


def PoissonGumbel2(lam,N):
	# calculate poisson prob.
	x = np.random.poisson(lam,N)

	# Generate Gumbel(0)
	yu = np.random.uniform(0,1,N)
	y = -np.log(-np.log(yu+1e-20)+1e-20)

	z = x+y
	return np.argmax(z)


# Define poisson parameter
lam = 2

# Define intervel
N= 50

iteration = 1000
l = np.zeros(iteration)
for i in range(iteration):
	z = PoissonGumbelMax(lam,N)
	l[i]= z



mean= np.mean(l)
var = np.var(l)
print(mean,var)