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

M_400 = 2*np.pi*grid_parameters[:,3]*grid_parameters[:,4]*400*cst.AU**2/cst.Mjup
M_400 = np.reshape(M_400,(np.size(M_400),1))
grid_parameters = np.hstack((grid_parameters,M_400))

"""
Define the limits of the grid
"""
def Sigma_max(R,M_star):
	return 0.2    * M_star*cst.Msun / (2*np.pi*R*400*cst.AU**2)

def Sigma_min(R,M_star):
	return 3.2e-5 * M_star*cst.Msun / (2*np.pi*R*400*cst.AU**2)

"""Parent class that returns the photoevaporation rate."""
class FRIEDInterpolator(object):

	def PE_rate(self, query_inputs, extrapolate=False):
		query_log = tuple(np.log10(query_inputs)) # Take logarithm of input values
		M_dot = self.M_dot_interp(query_log) # Perform the interpolation
		return np.power(10,M_dot) # Return exponentiated mass rate

"""Linear interpolators - using either disc mass (M), surface density (S)"""
"""
2D interpolators (no stellar mass or UV for computational speed) in log space
"""
class FRIED_2D(FRIEDInterpolator):
	def __init__(self, grid_parameters, grid_rate, M_star, UV, use_keys):
		self._Mstar = M_star
		self._UV = UV
		select_mass = (np.abs(grid_parameters[:,0] - M_star)<0.001) # Filter based on ones with the correct mass
		select_UV = (np.abs(grid_parameters[:,1] - UV)<0.001) # Filter based on ones with the correct UV
		select_MUV = select_mass * select_UV
		grid_inputs_2D = grid_parameters[select_MUV,:] # Apply filter
		self.selected_inputs = grid_inputs_2D[:,(use_keys[0],use_keys[1])]
		self.Sigma_inputs = grid_inputs_2D[:,(3,4)] # Select only the columns necessary - Sigma_disc, R_disc
		self.selected_rate = grid_rate[select_MUV]
		self.M_dot_interp = interpolate.LinearNDInterpolator(np.log10(self.selected_inputs),self.selected_rate) # Build interpolator on log of inputs

	def extrapolate_master(self,query_inputs,calc_rates):
		# At low surface densities and large enough radii, use scaling law M_dot \propto R \Sigma
		low_Sigma = ( query_inputs[0] < Sigma_min(query_inputs[1],self._Mstar) )
		ot_regime = low_Sigma * (calc_rates > 1e-10)
		#base_regime = low_Sigma * (calc_rates <= 1e-10)
                # By calling PE_rate with the maximum of Sigma, Sigma_min, just rescale
		scaling_factor = (query_inputs[0]/Sigma_min(query_inputs[1],self._Mstar))
		calc_rates[ot_regime] *= scaling_factor[ot_regime]

		# At high surface densities, clip to top of grid
		envelope_regime = ( query_inputs[0] > Sigma_max(query_inputs[1],self._Mstar) ) * (query_inputs[1] > 1) * (query_inputs[1] < 400)
                # By calling PE_rate with the minimum of Sigma, Sigma_max, no further calculations needed

		return ot_regime , envelope_regime, calc_rates

class FRIED_2DS(FRIED_2D):
	#Interpolates on surface density (S)
	def __init__(self, grid_parameters, grid_rate, M_star, UV):
		super().__init__(grid_parameters, grid_rate, M_star, UV, [3,4])
	#Extrapolation routine works here
	def extrapolate(self,query_inputs,calc_rates):
		return self.extrapolate_master(query_inputs,calc_rates)

class FRIED_2DM(FRIED_2D):
	# Interpolates on mass (M)
	def __init__(self, grid_parameters, grid_rate, M_star, UV):
		super().__init__(grid_parameters, grid_rate, M_star, UV, [2,4])
	#Extrapolation routine doesn't work here
	def extrapolate(self,query_inputs,calc_rates):
		print("Extrapolation not valid when interpolating on mass")

class FRIED_2DM400(FRIED_2D):
	# Interpolates on mass (M400)
	def __init__(self, grid_parameters, grid_rate, M_star, UV):
		super().__init__(grid_parameters, grid_rate, M_star, UV, [5,4])
	#Extrapolation routine doesn't work here
	def extrapolate(self,query_inputs,calc_rates):
		print("Extrapolation not valid when interpolating on mass")

class FRIED_2DMS(FRIED_2DM):
	# Interpolates on mass (M) but is provided with surface density (S)
	def PE_rate(self, query_inputs,extrapolate=True):
		new_query = np.array(query_inputs) # New array to hold modified query
		# Clip densities to ones in grid
		if extrapolate:
			re_Sigma = np.minimum(query_inputs[0], Sigma_max(query_inputs[1],self._Mstar))
			re_Sigma = np.maximum(re_Sigma, Sigma_min(query_inputs[1],self._Mstar))
		else:
			re_Sigma = query_inputs[0]
		# Convert sigma to a disc mass (for 1/R profile) and replace in query
		Mass_calc = 2*np.pi * re_Sigma * (query_inputs[1]*cst.AU)**2 / (cst.Mjup)
		new_query[0] = Mass_calc
		# Calculate rates
		calc_rates = super().PE_rate(new_query)
		# Adjust calculated rates according to extrapolation prescription
		if extrapolate:
			_, _, calc_rates = self.extrapolate(query_inputs,calc_rates)
		return calc_rates
	#Extrapolation routine works here
	def extrapolate(self,query_inputs,calc_rates):
		return self.extrapolate_master(query_inputs,calc_rates)

class FRIED_2DM400S(FRIED_2DM400):
	# Interpolates on mass at 400 AU (M400) but is provided with surface density (S)
	def PE_rate(self, query_inputs,extrapolate=True):
		new_query = np.array(query_inputs) # New array to hold modified query
		# Clip densities to ones in grid
		if extrapolate:
			re_Sigma = np.minimum(query_inputs[0], Sigma_max(query_inputs[1],self._Mstar))
			re_Sigma = np.maximum(re_Sigma, Sigma_min(query_inputs[1],self._Mstar))
		else:
			re_Sigma = query_inputs[0]
		# Convert sigma to a disc mass at 400 AU (for 1/R profile) and replace in query
		Mass_400 = 2*np.pi * re_Sigma * (query_inputs[1]*cst.AU) * (400*cst.AU) / (cst.Mjup)
		new_query[0] = Mass_400 # Replace first query parameter with mass
		# Calculate rates
		calc_rates =  super().PE_rate(new_query)
		# Adjust calculated rates according to extrapolation prescription
		if extrapolate:
			_, _, calc_rates = self.extrapolate(query_inputs,calc_rates)
		return calc_rates
	#Extrapolation routine works here
	def extrapolate(self,query_inputs,calc_rates):
		return self.extrapolate_master(query_inputs,calc_rates)

"""Not actually any faster"""
class FRIED_reggridded(object):
	def __init__(self, grid_parameters, grid_rate, M_star, UV, use_keys=[5,4]):
		select_mass = (np.abs(grid_parameters[:,0] - M_star)<0.001) # Filter based on ones with the correct mass
		select_UV = (np.abs(grid_parameters[:,1] - UV)<0.001) # Filter based on ones with the correct UV
		select_MUV = select_mass * select_UV
		grid_inputs_2D = grid_parameters[select_MUV,:] # Apply filter
		self.selected_inputs = grid_inputs_2D[:,(use_keys[0],use_keys[1])]
		self.Sigma_inputs = grid_inputs_2D[:,(3,4)] # Select only the columns necessary - Sigma_disc, R_disc
		self.selected_rate = grid_rate[select_MUV]
		radii = np.array([1,5,10,20,30,40,50,75,100,150,200,250,300,350,400])
		masses = np.array([3.2e-3,0.1,1.12,3.16,8.94,20])/100*M_star*cst.Msun/cst.Mjup
		input_rate = self.selected_rate.reshape(masses.size,radii.size)
		input_rate = input_rate[:,::-1]
		#input_rate = input_rate.transpose()
		self.M_dot_interp = interpolate.RegularGridInterpolator((np.log10(masses),np.log10(radii)),input_rate,bounds_error=False) # Build interpolator on log of inputs

	def PE_rate(self, query_inputs):
		Mass_400 = 2*np.pi * query_inputs[0] * (query_inputs[1]*cst.AU) * (400*cst.AU) / (cst.Mjup) # Convert sigma to a disc mass (for 1/R profile)
		new_query = np.log10(query_inputs)
		new_query[0] = np.log10(Mass_400)
		new_query = tuple(new_query)
		M_dot = self.M_dot_interp(new_query) # Perform the interpolation
		return np.power(10,M_dot) # Return exponentiated mass rate

"""
3D interpolators (no stellar mass for computational speed) in log space
"""
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
"""
4D linear space interpolators (need upgrading)
"""
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

"""# Function for comparing the interpolated values calculated here, with the ones from www.friedgrid.com/Tool, which must be manually downloaded and saved into a file. 
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
	plt.show()"""

def D2_space():
        M_star = 0.3
        UV = 100

        interp_type = '400'
        if (interp_type == 'MS'):
            photorate = FRIED_2DMS(grid_parameters,grid_rate,M_star,UV)
        elif (interp_type == 'S'):
            photorate = FRIED_2DS(grid_parameters,grid_rate,M_star,UV)
        elif (interp_type == '400'):
            photorate = FRIED_2DM400S(grid_parameters,grid_rate,M_star,UV)
        elif (interp_type == 'reg'):
            photorate = FRIED_reggridded(grid_parameters,grid_rate,M_star,UV)

        R = np.linspace(1,400,1600,endpoint=True)
        Sigma = np.logspace(-5,3,81)
        (R_interp, Sigma_interp) = np.meshgrid(R,Sigma)
        rates = photorate.PE_rate((Sigma_interp,R_interp))

        plt.rcParams['text.usetex'] = "True"
        import matplotlib.colors as colors
        pcm = plt.contourf(R_interp,Sigma_interp,rates,levels=np.logspace(-11,-4,15),norm=colors.LogNorm(vmin=1e-10,vmax=1e-4,clip=True))
        Sig_max = Sigma_max(R,M_star)
        Sig_min = Sigma_min(R,M_star)
        plt.plot(R,Sig_max,linestyle='--',color='red',label='$\Sigma_{max}$')
        plt.plot(R,Sig_min,linestyle='--',color='red',label='$\Sigma_{min}$')

        #grid_inputs_2D = photorate.Sigma_inputs
        #plt.plot(grid_inputs_2D[:,1],grid_inputs_2D[:,0],marker='x',color='black',linestyle='None')

        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('$R~/\mathrm{AU}$')
        plt.ylabel('$\Sigma~/\mathrm{g~cm}^{-2}$')
        plt.xlim([1,400])
        plt.ylim([1e-5,1e3])
        plt.colorbar(pcm, label='Mass Loss Rate ($M_\odot~\mathrm{yr}^{-1}$)')
        if (interp_type == 'MS'):
            plt.title("Interpolation on $M(\Sigma)$")
        elif (interp_type == 'S'):
            plt.title("Interpolation on $\Sigma$")
        elif (interp_type == '400'):
            plt.title("Interpolation on $M_{400}(\Sigma)$")
        elif (interp_type == 'reg'):
            plt.title("Interpolation on $M_{400}$")
        plt.savefig('Interp_limits_'+interp_type+'_'+str(M_star)+'C.png')
        
        
        fig, axes = plt.subplots(3,2,sharex='col',sharey='row')
        fig.subplots_adjust(hspace=0,wspace=0,top=0.92)
        axes=axes.flatten()
        plt.rcParams['text.usetex'] = "True"
        i_int = [1200,1000,800,600,400,200]
        j=0
        for i in i_int:
            #plt.subplot(3,2,j)
            plt.sca(axes[j])
            plt.loglog(Sigma,rates[:,i])
            first = np.where(rates[:,i] > 0)[0][0]
            M0 = rates[:,i][first]
            S0 = Sigma[first]
            plt.plot(Sigma,M0*(Sigma/S0),linestyle='-.',color='black')
            plt.plot(Sigma_min(R[i],M_star)*np.ones(2),[1e-11,1e-3],linestyle='--',color='red')
            plt.plot(Sigma_max(R[i],M_star)*np.ones(2),[1e-11,1e-3],linestyle='--',color='red')
            plt.ylim([1e-11,1e-3])
            plt.legend(labels=['$R={:.1f}~\mathrm{{AU}}$'.format(R[i])],loc=2)
            if (j==2):
                plt.ylabel('$\dot{M} /M_\odot\mathrm{yr}^{-1}$')
            j+=1

        fig.add_subplot(111, frameon=False)
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        plt.grid(False)
        plt.xlabel('$\Sigma /\mathrm{g~cm}^{-2}$')
        plt.suptitle('Dependence of $\dot{M}$ on $\Sigma$ at fixed radius')
        plt.savefig('Interp_FixedR_'+interp_type+'_'+str(M_star)+'C.png')  
                      
        #plt.show()

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
	#FRIED_Rates_2DM = FRIED_2DM(grid_parameters,grid_rate,1,100)

        D2_space()

