import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import DiscEvolution.constants as cst

class Formatter(object):
    def __init__(self, im):
        self.im = im
    def __call__(self, x, y):
        z = self.im.get_array()[int(y), int(x)]
        return 'x={:.01f}, y={:.01f}, z={:.01f}'.format(x, y, z)

## Import data from FRIED table (Haworth et al 2018)
## Data listed as M_star, UV, M_disc, Sigma_disc, R_disc, M_dot
## Values given in linear space apart from M_dot which is given as its base 10 logarithm

# Take M_star, UV, M_disc, Sigma_disc, R_disc to build parameter space
grid_parameters = np.loadtxt(os.environ['DISC_CODE_ROOT']+'/FRIED/friedgrid.dat',skiprows=1,usecols=(0,1,2,3,4))
# Import M_dot
grid_rate = np.loadtxt(os.environ['DISC_CODE_ROOT']+'/FRIED/friedgrid.dat',skiprows=1,usecols=5)
#grid_rate_exp = np.power(10,grid_rate)

# Calculate mass wihtin 400 AU and add to grid as column 6
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

class FRIED_2DM400M(FRIED_2DM400):
	# Interpolates on mass at 400 AU (M400) but is provided with surface density (M)
	def PE_rate(self, query_inputs,extrapolate=False):
		new_query = np.array(query_inputs) # New array to hold modified query
		# Convert to a disc mass at 400 AU (for 1/R profile) and replace in query
		Mass_400 = query_inputs[0] * (400 / query_inputs[1])
		new_query[0] = Mass_400 # Replace first query parameter with mass
		# Calculate rates
		calc_rates =  super().PE_rate(new_query)
		return calc_rates
	#Extrapolation routine works here
	def extrapolate(self,query_inputs,calc_rates):
		return self.extrapolate_master(query_inputs,calc_rates)

def D2_space(interp_type = '400', extrapolate=True, UV=1000, save=True):
        # Configure external parameters 
        M_star = 1.0
        #UV = 1000

        # Setup interpolator
        if (interp_type == 'MS'):
            photorate = FRIED_2DMS(grid_parameters,grid_rate,M_star,UV)
        elif (interp_type == 'S'):
            photorate = FRIED_2DS(grid_parameters,grid_rate,M_star,UV)
        elif (interp_type == '400'):
            photorate = FRIED_2DM400S(grid_parameters,grid_rate,M_star,UV)

        # Setup interpolation grid
        R = np.linspace(1,400,1600,endpoint=True)
        Sigma = np.logspace(-5,3,81)
        (R_interp, Sigma_interp) = np.meshgrid(R,Sigma)

        # Interpolate
        rates = photorate.PE_rate((Sigma_interp,R_interp))

        # Prepare for plotting
        plt.rcParams['text.usetex'] = "True"
        plt.rcParams['font.family'] = "serif"
        import matplotlib.colors as colors
        fig = plt.figure()

        # Plot
        n_levels = 15*(2-save) 
        pcm = plt.contourf(R_interp,Sigma_interp,rates,levels=np.logspace(-11,-4,n_levels),norm=colors.LogNorm(vmin=1e-11,vmax=1e-4,clip=True),extend='min')
        Sig_max = Sigma_max(R,M_star)
        Sig_min = Sigma_min(R,M_star)
        plt.plot(R,Sig_max,linestyle='--',color='red',label='$\Sigma_{max}$')
        plt.plot(R,Sig_min,linestyle='--',color='red',label='$\Sigma_{min}$')

        #grid_inputs_2D = photorate.Sigma_inputs
        #plt.plot(grid_inputs_2D[:,1],grid_inputs_2D[:,0],marker='x',color='black',linestyle='None')

        # Adorn plot
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('$R~/\mathrm{AU}$',fontsize=18)
        plt.ylabel('$\Sigma~/\mathrm{g~cm}^{-2}$', fontsize=18)
        plt.tick_params(axis='x', which='major', labelsize=14)
        plt.tick_params(axis='y', which='major', labelsize=14)
        plt.xlim([1,400])
        plt.ylim([1e-5,1e3])
        bar_label='Mass Loss Rate ($M_\odot~\mathrm{yr}^{-1}$)'
        cbar = plt.colorbar(pcm)
        cbar.ax.tick_params(labelsize=14) 
        cbar.set_label(label=bar_label, fontsize=18)

        if False:
            M_disc = 0.02
            R_C = 100
            Sig_profile = M_disc * cst.Msun / (2*np.pi * R_C * R * cst.AU**2) * np.exp(-R/R_C)
            plt.plot(R,Sig_profile,color='orange',linestyle=':')
            R_C = 10
            Sig_profile = M_disc * cst.Msun / (2*np.pi * R_C * R * cst.AU**2) * np.exp(-R/R_C)
            plt.plot(R,Sig_profile,color='orange',linestyle=':')
            Sig_profile = M_disc * cst.Msun / (2*np.pi * 400 * R * cst.AU**2)
            plt.plot(R,Sig_profile,color='orange',linestyle=':')

        if save:
            """if (interp_type == 'MS'):
                plt.title("Interpolation on $M(\Sigma)$",fontsize=24)
            elif (interp_type == 'S'):
                plt.title("Interpolation on $\Sigma$",fontsize=24)
            elif (interp_type == '400'):
                plt.title("Interpolation on $M_{400}(\Sigma)$",fontsize=24)
            elif (interp_type == 'reg'):
                plt.title("Interpolation on $M_{400}$",fontsize=24)"""
            plt.tight_layout()
            plt.savefig('Interpolation_'+interp_type+'_'+str(UV)+'.png')
            plt.show()
        else:
            return fig

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="None")
    parser.add_argument("--index", "-i", type=int, nargs='+', default=0)
    parser.add_argument("--FUV", "-u", type=float, default=1000)
    args = parser.parse_args()

    model = args.model
    indices = np.array(args.index)

    if (model=="None"):
        D2_space(interp_type='400', extrapolate=True, UV=args.FUV)
    else:
        import json
        from KeyScripts import run_model
        from scripts.snap_reader import DiscReader
        #D2_space(interp_type='400',extrapolate=True,save=False)
        model_params = json.load(open(model, 'r'))
        DIR = model_params['output']['directory']

        inputdata = np.loadtxt(DIR+"/"+model_params['output']['plot_name']+"_discproperties.dat")
        plot_times = inputdata[:,0]
        R_wind = inputdata[:,2]
        #disc, _, _, _, _, _, Dt_nv = run_model.setup_wrapper(model_params)

        for i in indices:
            fig_i = D2_space(interp_type='400',extrapolate=True,save=False)
            plt.figure(fig_i.number)

            reader = DiscReader(DIR, 'disc')
            disc = reader[i]
            i_wind = np.argmin(np.abs(R_wind[i]-disc.R))

            plt.plot(disc.R,disc.Sigma,linestyle=':',color='orange')
            plt.plot(disc.R[i_wind],disc.Sigma[i_wind],marker='x',color='orange')

            cur_t = plot_times[i]
            if (cur_t>0):    
                logt = np.log10(cur_t)
                exponent = int(logt)
                prefactor = np.power(10,logt-exponent)
                plt.title('$t={:.2f}\\times10^{:d}~\mathrm{{yr}}$'.format(prefactor,exponent), fontsize=18)
            else:
                plt.title('$t=0~\mathrm{{yr}}$', fontsize=18)
            plt.savefig('profile_trace_{}.pdf'.format(i))
            #plt.show()

