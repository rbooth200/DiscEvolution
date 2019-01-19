import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import os
import sys
import DiscEvolution.constants as cst

## Import data from FRIED table (Haworth et al 2018)
## Data listed as M_star, UV, M_disc, Sigma_disc, R_disc, M_dot
## Values given in linear space apart from M_dot which is given as its base 10 logarithm

# Take M_star, UV, M_disc, Sigma_disc, R_disc to build parameter space
grid_parameters = np.loadtxt(os.environ['DISC_CODE_ROOT']+'/FRIED/friedgrid.dat',skiprows=1,usecols=(0,1,2,3,4))
# Import M_dot
grid_rate = np.loadtxt(os.environ['DISC_CODE_ROOT']+'/FRIED/friedgrid.dat',skiprows=1,usecols=5)
#grid_rate_exp = np.power(10,grid_rate)

"""Parent class that returns the photoevaporation rate."""
class FRIEDInterpolator(object):

	def PE_rate(self, query_inputs):
		query_log = tuple(np.log10(query_inputs)) # Take logarithm of input values
		M_dot = self.M_dot_interp(query_log) # Perform the interpolation
		return np.power(10,M_dot) # Return exponentiated mass rate

"""Linear interpolators - using either disc mass (M), surface density (S)"""
# 3D interpolators (no stellar mass for computational speed) in log space
class FRIED_3DM(FRIEDInterpolator):
	# Interpolates on mass (M)

	def __init__(self, grid_parameters, grid_rate, M_star):
		select_mass = (np.abs(grid_parameters[:,0] - M_star)<0.001) # Filter based on ones with the correct mass
		grid_inputs_3D = grid_parameters[select_mass,:] # Apply filter
		grid_inputs_3DM = grid_inputs_3D[:,(1,2,4)] # Select only the columns necessary - UV, M_disc, R_disc
		self.M_dot_interp = interpolate.LinearNDInterpolator(np.log10(grid_inputs_3DM),grid_rate[select_mass]) # Build interpolator on log of inputs 

class FRIED_3DS(FRIEDInterpolator):
	#Interpolates on surface density (S)

	def __init__(self, grid_parameters, grid_rate, M_star):
		select_mass = (np.abs(grid_parameters[:,0] - M_star)<0.001) # Filter based on ones with the correct mass
		grid_inputs_3D = grid_parameters[select_mass,:] # Apply filter
		grid_inputs_3DS = grid_inputs_3D[:,(1,3,4)] # Select only the columns necessary - UV, Sigma_disc, R_disc
		self.M_dot_interp = interpolate.LinearNDInterpolator(np.log10(grid_inputs_3DS),grid_rate[select_mass]) # Build interpolator on log of inputs

class FRIED_3DMS(FRIED_3DM):
	# Interpolates on mass but is provided with surface density

	def PE_rate(self, query_inputs):
		Mass_calc = 2*np.pi * query_inputs[1] * (query_inputs[2]*cst.AU)**2 / (cst.Mjup) # Convert sigma to a disc mass (for 1/R profile)
		new_query = np.log10(query_inputs)
		new_query[1] = np.log10(Mass_calc)
		new_query = tuple(new_query)
		M_dot = self.M_dot_interp(new_query) # Perform the interpolation
		return np.power(10,M_dot) # Return exponentiated mass rate

# 2D interpolators (no stellar mass or UV for computational speed) in log space
class FRIED_2DM(FRIEDInterpolator):
	# Interpolates on mass (M)

	def __init__(self, grid_parameters, grid_rate, M_star, UV):
		select_mass = (np.abs(grid_parameters[:,0] - M_star)<0.001) # Filter based on ones with the correct mass
		select_UV = (np.abs(grid_parameters[:,1] - UV)<0.001) # Filter based on ones with the correct UV
		select_MUV = select_mass * select_UV
		grid_inputs_2D = grid_parameters[select_MUV,:] # Apply filter
		grid_inputs_2DM = grid_inputs_2D[:,(2,4)] # Select only the columns necessary - UV, M_disc, R_disc
		self.M_dot_interp = interpolate.LinearNDInterpolator(np.log10(grid_inputs_2DM),grid_rate[select_MUV]) # Build interpolator on log of inputs 

class FRIED_2DS(FRIEDInterpolator):
	#Interpolates on surface density (S)

	def __init__(self, grid_parameters, grid_rate, M_star, UV):
		select_mass = (np.abs(grid_parameters[:,0] - M_star)<0.001) # Filter based on ones with the correct mass
		select_UV = (np.abs(grid_parameters[:,1] - UV)<0.001) # Filter based on ones with the correct UV
		select_MUV = select_mass * select_UV
		grid_inputs_2D = grid_parameters[select_MUV,:] # Apply filter
		grid_inputs_2DS = grid_inputs_2D[:,(3,4)] # Select only the columns necessary - UV, Sigma_disc, R_disc
		self.M_dot_interp = interpolate.LinearNDInterpolator(np.log10(grid_inputs_2DS),grid_rate[select_MUV]) # Build interpolator on log of inputs

class FRIED_2DMS(FRIED_2DM):
	# Interpolates on mass but is provided with surface density

	def PE_rate(self, query_inputs):
		Mass_calc = 2*np.pi * query_inputs[0] * (query_inputs[1]*cst.AU)**2 / (cst.Mjup) # Convert sigma to a disc mass (for 1/R profile)
		new_query = np.log10(query_inputs)
		new_query[0] = np.log10(Mass_calc)
		new_query = tuple(new_query)
		M_dot = self.M_dot_interp(new_query) # Perform the interpolation
		return np.power(10,M_dot) # Return exponentiated mass rate

# 4D linear space interpolators (need upgrading)	
'''
class FRIED_4DM(FRIEDInterpolator):

	def __init__(self, grid_parameters, grid_rate_exp):
		grid_inputs_4DM = grid_parameters[:,(0,1,2,4)]
		self.M_dot_interp = interpolate.LinearNDInterpolator(grid_inputs_4DM,grid_rate_exp)

class FRIED_4DM_log(FRIEDInterpolator):

	def __init__(self, grid_parameters, grid_rate):
		grid_inputs_4DM = grid_parameters[:,(0,1,2,4)]
		grid_inputs_log = np.stack((grid_parameters[:,0], np.log10(grid_parameters[:,1]), np.log10(grid_parameters[:,2]), np.log10(grid_parameters[:,4]) ),-1)
		self.M_dot_interp = interpolate.LinearNDInterpolator(grid_inputs_log,grid_rate)

class FRIED_4DS(FRIEDInterpolator):

	def __init__(self, grid_parameters, grid_rate_exp):
		grid_inputs_4DS = grid_parameters[:,(0,1,3,4)]
		self.M_dot_interp = interpolate.LinearNDInterpolator(grid_inputs_4DS,grid_rate_exp)
'''

'''
What follows are functions used in the testing of the above routines
They are not designed for implementation in the code.
'''

# Function for comparing the interpolated values calculated here, with the ones from www.friedgrid.com/Tool, which must be manually downloaded and saved into a file. 
def compareinterp(query_inputs=(1.0,1.2,0.1066,123),downloadedrates='FRIEDinterp.dat'):
	FRIED_interp = np.loadtxt(downloadedrates)
	
	x = np.power(10,FRIED_interp[:,0])
	a = np.zeros_like(x)
	b = np.zeros_like(x)
	c = np.zeros_like(x)
	d = np.zeros_like(x)
	e = np.zeros_like(x)

	FRIED_Rates_3DM = FRIED_3DM(grid_parameters,grid_rate,query_inputs[0])
	#FRIED_Rates_4DM_log = FRIED_4DM_log(grid_parameters,grid_rate)
	FRIED_Rates_3DS = FRIED_3DMS(grid_parameters,grid_rate,query_inputs[0])

	for i in range(len(x)):
		query = (x[i],query_inputs[1],query_inputs[3])
		query2 = (x[i],query_inputs[2],query_inputs[3])
		#a[i] = FRIED_Rates_4DM_log.PE_rate((query_inputs[0],np.log10(x[i]),np.log10(query_inputs[1]),np.log10(query_inputs[3])))
		#b[i] = np.log10(PE_rate_4DS(*query2))
		c[i] = FRIED_Rates_3DM.PE_rate(query)
		d[i] = FRIED_Rates_3DS.PE_rate(query2)

	plt.rcParams['text.usetex'] = "True"
	#plt.plot(FRIED_interp[:,0],a,color='green',linestyle='-',marker='+', label='4D, Mass, log')
	#plt.plot(FRIED_interp[:,0],b,color='red',linestyle='-',marker='+', label='4D, Density')
	plt.semilogy(FRIED_interp[:,0],c,color='green',linestyle='-',marker='x', label='3D, Mass')
	plt.semilogy(FRIED_interp[:,0],d,color='red',linestyle='-',marker='x', label='3D, Density')
	plt.semilogy(FRIED_interp[:,0],np.power(10,FRIED_interp[:,1]),color='blue',linestyle='-',marker='x',label='friedgrid.com Tool')
	plt.xlabel('UV Field ($G_0$)')
	plt.ylabel('Mass loss rate, $\log(\dot{M}/M_\odot\mathrm{yr}^{-1})$')
	plt.title('Interpolation for $M_*={}$, $M_{{d}}={}$, $\Sigma_G={}$, $R_{{d}}={}$'.format(*query_inputs))
	plt.legend()
	plt.savefig('InterpolationComparison.png')
	#plt.show()

def Fdependence(downloadedrates='FRIEDinterp5.dat'):
	FRIED_interp = np.loadtxt(downloadedrates)
	#Take useful bit 9-22
	Useful = FRIED_interp[9:23,:]
	UsefulT = np.transpose(Useful)
	polyfitted = np.polyfit(UsefulT[0],UsefulT[1],1)
	print (polyfitted)
	fittedvals = np.polyval(polyfitted, UsefulT[0])
	plt.plot(UsefulT[0],UsefulT[1],marker='x')
	plt.plot(UsefulT[0],fittedvals)
	plt.show()

if __name__ == "__main__":
	#query_inputs = (float(sys.argv[1]),float(sys.argv[2]),float(sys.argv[3]),float(sys.argv[4]))
	#sigma_R = float(sys.argv[3]) * 1.9e30 / (2*np.pi*(float(sys.argv[4])*1.5e13)**2)
	#query_2 = (float(sys.argv[1]),float(sys.argv[2]),sigma_R,float(sys.argv[4]))
	
	#print(PE_rate_4DM(*query_inputs))
	#print(PE_rate_4DS(*query_2))
	#print(PE_rate_3DM(*query_inputs))
	#print(PE_rate_3DS(*query_2))
	
	#testinterp = FRIED_3DM(grid_parameters,grid_rate_exp,float(sys.argv[1]))
	#print(testinterp.PE_rate((float(sys.argv[2]),float(sys.argv[3]),float(sys.argv[4]))))
	
	#compareinterp((1.0,7.41,1,100),'FRIEDinterp4.dat')
	#Fdependence('FRIEDinterp6.dat')
	FRIED_Rates_2DM = FRIED_2DM(grid_parameters,grid_rate,1,100)
