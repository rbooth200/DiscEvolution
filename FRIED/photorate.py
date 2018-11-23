import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import os

# Import data from FRIED table (Haworth et al 2018)
# Data listed as M_star, UV, M_disc, Sigma_disc, R_disc, M_dot
# Take M_star, UV, M_disc, Sigma_disc, R_disc to build parameter space
grid_parameters = np.loadtxt(os.environ['DISC_CODE_ROOT']+'/FRIED/friedgrid.dat',skiprows=1,usecols=(0,1,2,3,4))
# Import M_dot
grid_rate = np.loadtxt(os.environ['DISC_CODE_ROOT']+'/FRIED/friedgrid.dat',skiprows=1,usecols=5)

# Function to return the mass loss rate by interpolating the mass loss rate on the UV flux, disc mass and disc radius for a stellar mass in the grid (FAST, but restrictive)
# Acceptable masses are 0.05, 0.1, 0.3, 0.5, 0.8, 1.0, 1.3, 1.6, 1.9 times solar
def PE_rate_M(M_star = 1.0, UV = 10, M_disc = 1, R_disc = 400):
	grid_inputs = grid_parameters[:,(1,2,4)]
	query_inputs = (UV, M_disc, R_disc)
	grid_rate_exp = np.power(10,grid_rate)

	select_mass = (np.abs(grid_parameters[:,0] - M_star)<0.001)
	if (np.sum(select_mass)==0):
		print("Invalid Mass")		
		return -1
	else:
		M_dot = interpolate.griddata(grid_inputs[select_mass,:],grid_rate_exp[select_mass],query_inputs,method='linear')
		return M_dot

# Function to return the mass loss rate by interpolating the mass loss rate on the UV flux, disc mass and disc radius and stellar mass (SLOW)
# Currently disagreeing with above for reasons unknown
def PE_rate_M_S(M_star = 1.0, UV = 10, M_disc = 1, R_disc = 400):
	grid_inputs = grid_parameters[:,(0,1,2,4)]
	query_inputs = (M_star, UV, M_disc, R_disc)
	grid_rate_exp = np.power(10,grid_rate)

	M_dot = interpolate.griddata(grid_inputs,grid_rate_exp,query_inputs,method='linear')
	return M_dot

# Function to return the mass loss rate by interpolating the mass loss rate on the UV flux, gas surface density and disc radius for a stellar mass in the grid (FAST, but restrictive)
# Acceptable masses are 0.05, 0.1, 0.3, 0.5, 0.8, 1.0, 1.3, 1.6, 1.9 times solar
def PE_rate_D(M_star = 1.0, UV = 10, Sigma_G = 1, R_disc = 400):
	grid_inputs = grid_parameters[:,(1,3,4)]
	query_inputs = (UV, Sigma_G, R_disc)
	grid_rate_exp = np.power(10,grid_rate)

	select_mass = (np.abs(grid_parameters[:,0] - M_star)<0.001)
	if (np.sum(select_mass)==0):
		print("Invalid Mass")		
		return -1
	else:
		M_dot = interpolate.griddata(grid_inputs[select_mass,:],grid_rate_exp[select_mass],query_inputs,method='linear')
		return M_dot

# Function for comparing the interpolated values calculated here, with the ones from www.friedgrid.com/Tool, which must be manually downloaded and saved into a file. 
def compareinterp(query_inputs=(1.0,1.2,123),downloadedrates='FRIEDinterp.dat'):
	FRIED_interp = np.loadtxt(downloadedrates)
	
	x = np.power(10,FRIED_interp[:,0])
	z = np.zeros_like(x)
	for i in range(len(x)):
		query = (query_inputs[0],x[i],query_inputs[1],query_inputs[2])
		z[i] = np.log10(PE_rate(*query))	

	plt.rcParams['text.usetex'] = "True"
	plt.plot(FRIED_interp[:,0],z,color='green',linestyle='-',marker='x', label='Calculated here')
	plt.plot(FRIED_interp[:,0],FRIED_interp[:,1],color='blue',linestyle='-',marker='x',label='friedgrid.com Tool')
	plt.xlabel('UV Field ($G_0$)')
	plt.ylabel('Mass loss rate, $\log(\dot{M})$')
	plt.title('Interpolation for $M_*={}$, $M_{{d}}={}$, $R_{{d}}={}$'.format(*query_inputs))
	plt.legend()
	plt.show()

if __name__ == "__main__":
	import sys
	query_inputs = (float(sys.argv[1]),float(sys.argv[2]),float(sys.argv[3]),float(sys.argv[4]))
	print(PE_rate_M(*query_inputs))
	
	#print(PE_rate_M_S(*query_inputs))
	
	#sigma_R = float(sys.argv[3]) * 1.9e30 / (2*np.pi*(float(sys.argv[4])*1.5e13)**2)
	#query_2 = (float(sys.argv[1]),float(sys.argv[2]),sigma_R,float(sys.argv[4]))
	#print(PE_rate_D(*query_2))

	#compareinterp()
