# run_model.py
#
# Author: R. Booth
# Date: 4 - Jun - 2018
#
# Run a disc evolution model with transport and absorption / desorption but
# no other chemical reactions. 
#
# Note:
#   The environment variable 
#       "KROME_PATH=/home/rab200/WorkingCopies/krome_ilee/build"
#   should be set.
###############################################################################
import os
import json
import numpy as np
import scipy as scp
import matplotlib.pyplot as plt
import copy
from DiscEvolution.constants import Msun, AU, yr, Mjup
from DiscEvolution.grid import Grid
from DiscEvolution.star import SimpleStar
from DiscEvolution.eos  import IrradiatedEOS, LocallyIsothermalEOS
from DiscEvolution.dust import DustGrowthTwoPop
from DiscEvolution.opacity import Tazzari2016
from DiscEvolution.viscous_evolution import ViscousEvolution
from DiscEvolution.disc import AccretionDisc
from DiscEvolution.dust import SingleFluidDrift
from DiscEvolution.diffusion import TracerDiffusion
from DiscEvolution.driver import DiscEvolutionDriver
from DiscEvolution.io import Event_Controller
from DiscEvolution.disc_utils import mkdir_p
import DiscEvolution.photoevaporation as photoevaporation
import FRIED.photorate as photorate
import subprocess

###############################################################################
# Global Constants
###############################################################################
DefaultModel = "DiscConfig_default.json"

###############################################################################
# Global Functions
###############################################################################
def setup_disc(model):
    '''Create disc object from initial conditions'''
    # Setup the grid, star and equation of state
    p = model['grid']
    grid = Grid(p['R0'], p['R1'], p['N'], spacing=p['spacing'])

    p = model['star']
    star = SimpleStar(M=p['mass'], R=p['radius'], T_eff=p['T_eff'])
    
    p = model['eos']
    if p['type'] == 'irradiated':
        assert p['opacity'] == 'Tazzari2016'
        kappa = Tazzari2016()
        eos = IrradiatedEOS(star, model['disc']['alpha'], kappa=kappa)
    elif p['type'] == 'iso':
        eos = LocallyIsothermalEOS(star, p['h0'], p['q'], 
                                   model['disc']['alpha'])
    else:
        raise ValueError("Error: eos::type not recognised")
    eos.set_grid(grid)

    # Setup the physical part of the disc
    p = model['disc']
    if (('profile' in model['disc']) == False):
        Sigma = np.exp(-grid.Rc / p['Rc']) / (grid.Rc) # Lynden-Bell & Pringle
    elif (model['disc']['profile'] == 'LBP'):
        Sigma = np.exp(-grid.Rc / p['Rc']) / (grid.Rc) # Lynden-Bell & Pringle
    else:
        Sigma = 1.0 / (grid.Rc) # Power Law
    Sigma *= p['mass'] / np.trapz(Sigma, np.pi*grid.Rc**2)
    Sigma *= Mjup / AU**2

    eos.update(0, Sigma)

    try:
        feedback = model['disc']['feedback']
    except KeyError:
        feedback = True

    # If non-zero dust, set up a two population model, else use a simple accretion disc
    if (model['disc']['d2g']>0):
        disc = DustGrowthTwoPop(grid, star, eos, p['d2g'], Sigma=Sigma, Sc=model['disc']['Schmidt'], feedback=feedback, uf_ice=model['dust']['ice_frag_v'])
    else:
        disc = AccretionDisc(grid, star, eos, Sigma=Sigma)

    # Setup the UV irradiation
    p = model['uv']
    disc.set_UV(p['uv_field'])

    return disc


def setup_model(model, disc):
    '''Setup the physics of the model'''
    
    gas       = None
    dust      = None
    diffuse   = None
    chemistry = None

    if model['transport']['gas']:
        gas = ViscousEvolution(boundary='Mdot') ### NB this was changed !!!
    if model['transport']['diffusion']:
        diffuse = TracerDiffusion(Sc=model['disc']['Schmidt'])
    if model['transport']['radial drift']:
        dust = SingleFluidDrift(diffuse)
        diffuse = None

    # Inititate the correct photoevaporation routine
    if (model['uv']['photoevaporation'] == "FRIED" and disc.UV>0):
        photoevap = photoevaporation.FRIEDExternalEvaporationMS(disc) # Using 2DMS at 400
    elif (model['uv']['photoevaporation'] == "Constant"):
        photoevap = photoevaporation.FixedExternalEvaporation(disc, Mdot=1e-9)
    elif (model['uv']['photoevaporation'] == "compare"):
        photoevap = photoevaporation.FRIEDExternalEvaporationMS(disc) # Using 2DMS at 400
    elif (model['uv']['photoevaporation'] == "Integrated"):
        photoevap = photoevaporation.FRIEDExternalEvaporationM(disc) # Using integrated M(<R)
    else:
        photoevap = None

    return DiscEvolutionDriver(disc, 
                               gas=gas, dust=dust, diffusion=diffuse, photoevaporation=photoevap)
    
def setup_output(model):
    
    out = model['output']

    # Setup of the output controller
    """output_times = np.arange(out['first'], out['last'], out['interval'])
    if not np.allclose(out['last'], output_times[-1], 1e-12):
        output_times = np.append(output_times, out['last'])

    output_times *= yr"""

    # Logarithmic version
    first_log = np.log10(out['first'])
    last_log  = np.log10(out['last'])
    output_times = np.logspace(first_log,last_log,10*(last_log-first_log)+1,endpoint=True,base=10,dtype=int) * yr
    output_times = np.insert(output_times,0,0)

    # Setup of the plot controller
    if out['plot']:
        plot = np.array(out["plot_times"]) * yr
    else:
        plot = []

    EC = Event_Controller(save=output_times, plot=plot)
    
    # Base string for output:
    mkdir_p(out['directory'])
    mkdir_p(out['directory']+'_profiles')
    base_name = os.path.join(out['directory'], out['base'] + '_{:04d}')

    format = out['format']
    if format.lower() == 'hdf5':
        base_name += '.h5'
    elif format.lower() == 'ascii':
        base_name += '.dat'
    else:
        raise ValueError ("Output format {} not recognized".format(format))

    return base_name, EC
    
    
def run(model, io, base_name, plot_name, ylims, mass_loss_mode, dust_radii_thresholds, all_in, verbose=True, n_print=1000):
    i=-1 # Initialise the plotting number
    while not io.finished():
        ti = io.next_event_time()
        while model.t < ti:
            """The model breaks down when all at base rate because i_max = i_edge, Sigma(i_edge -> 0). Instead, terminate the model"""
            not_empty = (model.disc.Sigma_G > 0)
            if (model.photoevap is not None):
                Mdot = model.photoevap.mass_loss_rate(model.disc,not_empty)
                if (np.amax(Mdot)<=1e-10):
                    last_save=0
                    last_plot=0
                    if (np.size(io.event_times('save'))>0):
                        last_save = io.event_times('save')[-1]
                    if (np.size(io.event_times('plot'))>0):
                        last_plot = io.event_times('plot')[-1]
                    last_t = max(last_save,last_plot)
                    io.pop_events(last_t)
                    break

            dt = model(ti)

            if verbose and (model.num_steps % n_print) == 0:
                print('Nstep: {}'.format(model.num_steps))
                print('Time: {} yr'.format(model.t / yr))
                print('dt: {} yr'.format(dt / yr))

        if io.check_event(model.t, 'save'):
            if base_name.endswith('.h5'):
                model.dump_hdf5(base_name.format(io.event_number('save')))
            else:
                model.dump_ASCII(base_name.format(io.event_number('save')))

        if io.check_event(model.t, 'plot'):
            err_state = np.seterr(all='warn')

            print('Nstep: {}'.format(model.num_steps))
            print('Time: {} yr'.format(model.t / (2 * np.pi)))
            
            grid = model.disc.grid

            # Plot of initial evolutionary timescales
            if (i < 0 and isinstance(model.disc,DustGrowthTwoPop) and model.photoevap is not None):
                plt.figure()
                plt.rcParams['text.usetex'] = "True"
                plt.rcParams['font.family'] = "serif"

                # Photoevaporation calculations
                (M_evap, _, _, t_evap) = model.photoevap.get_timescale(model.disc)
                plt.loglog(grid.Rc, t_evap  / (2*np.pi), color='green', label='$t_{evap}$')

                # Radial drift timescale for fragmentation limited dust
                t_drift_f = 3 * model.disc.alpha * model.disc.star.M**(1/2) / (0.37 * 11/4 * model.disc._uf**2)
                plt.loglog(grid.Rc, t_drift_f * grid.Rc**(1/2) / (2*np.pi), color='red', label='$\\tau_{drift}$')

                # Growth timescale
                eps_tot = model.disc.dust_frac.sum(0)
                eps_tot = np.minimum(eps_tot,1.0)
                t_grow_e = model.disc._t_grow(eps_tot)

                # Work out local dust size limit, not including growth
                afrag_t = model.disc._frag_limit()
                adrift, afrag_d =  model.disc._drift_limit(eps_tot)
                afrag = np.minimum(afrag_t, afrag_d)
                a1    = np.minimum(afrag, adrift)
                # Growth time to maximum size is
                t_grow_max = t_grow_e * np.log(a1/model.disc._monomer)
                plt.loglog(grid.Rc, t_grow_max / (2*np.pi), color='blue', label='$t_{grow,max}$')

                # Growth time to entrained size
                a_ent = photoevaporation.Facchini_limit(model.disc, M_evap *(yr/Msun))
                t_grow_ent = t_grow_e * np.log(a_ent/model.disc._monomer)
                plt.loglog(grid.Rc, t_grow_ent / (2*np.pi), color='blue', linestyle='--', label='$t_{grow,ent}$')

                # Growth size to significantly drifting size
                F = model.disc.H / np.sqrt(model.disc.H**2+model.disc.R**2)
                a_rd = 2**(7/2) / (np.pi**(3/2) * 11/4) * F * (model.disc.R * model.disc.Omega_k / model.disc.cs)**3 * a_ent
                t_grow_rd = t_grow_e * np.log(a_rd/model.disc._monomer)
                plt.loglog(grid.Rc, t_grow_rd / (2*np.pi), color='blue', linestyle='-.', label='$t_{grow,rd}$')                

                # Label plot
                plt.legend()                
                plt.xlim([1,400])
                plt.xlabel('$R~/~\mathrm{AU}$',fontsize=13)
                plt.ylim([1e3,1e6])
                plt.ylabel('$t~/~\mathrm{yr}$', fontsize=13)
                plt.title('Initial Timescales of Dust Evolution', fontsize=16)
                plt.savefig(plot_name+"_timescales.png")
                
                # Print summary information
                when_drain = np.argmin(np.abs(t_evap-t_grow_rd))
                Rc_d = grid.Rc[when_drain]
                t_d = t_evap[when_drain]/(2*np.pi)
                t_drift_d = t_drift_f*Rc_d**(1/2)/(2*np.pi)
                print ("Dust will radially drift away from barrier when \t t = {} yr".format(t_d))
                print ("At this point the base of the wind is at \t\t R_w = {} AU".format(Rc_d))
                print ("This dust radially drifts with timescale \t\t t = {} yr".format(t_drift_d))
                print ("Hence the disc has drained after \t\t\t t = {} yr".format(t_d + 2 * t_drift_d))
                timescale_file = open(plot_name+"_timescales.dat",'w')
                timescale_file.write("Dust will radially drift away from barrier when  t = {} yr \n".format(t_d))
                timescale_file.write("At this point the base of the wind is at \t\t R_w = {} AU \n".format(Rc_d))
                timescale_file.write("This dust radially drifts with timescale \t\t t = {} yr \n".format(t_drift_d))
                timescale_file.write("Hence the disc has drained after \t\t\t\t t = {} yr \n".format(t_d + 2 * t_drift_d))
                timescale_file.close()

            # Set plot number and setup figure
            i+=1
            no_plots = 2 - int(model.photoevap == None) + int(isinstance(model.disc,DustGrowthTwoPop)) + int(model.dust != None)
            plt.subplots(no_plots,1,sharex=True,figsize=(8,2.5*no_plots))
            plt.rcParams['text.usetex'] = "True"
            plt.rcParams['font.family'] = "serif"
            
            ### SUBPLOT 1 ###
            # Plot of density profile
            plt.subplot(no_plots,1,1)
            plt.loglog(grid.Rc, model.disc.Sigma_G, label='$\Sigma_\mathrm{G}$')
            if (isinstance(model.disc,DustGrowthTwoPop)):
                plt.loglog(grid.Rc, model.disc.Sigma_D.sum(0), label='$\Sigma_\mathrm{D}$', linestyle='--')
            plt.ylabel('$\Sigma~/~\mathrm{g~cm}^{-2}$',fontsize=13)
            plt.xlim([1,500])
            plt.ylim(ylims)
            plt.legend(loc=3)

            ### SUBPLOT 2 ###
            # Plot of weighted mass loss rate profiles and comparisons
            if (model.photoevap is not None):
                plt.subplot(no_plots,1,2)
                cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

                # Get the raw mass loss rates 
                (M_evap, M_ann) = model.photoevap.unweighted_rates(model.disc)
                # Get the weighted mass loss rates (Track=True to save integrated quantity for plotting)
                (M_dot, _) = model.photoevap.optically_thin_weighting(model.disc, Track=True)

                # Mass profile on the left axes
                lha = plt.gca()
                lha.loglog(grid.Rc, M_ann, label='Annulus Mass')
                lha.set_ylabel('$M~/~\mathrm{g}$',fontsize=13)
                if (i==0):
                    ylimsl = lha.get_ylim()
                lha.set_ylim(ylimsl)
                if (no_plots == 2):
                    plt.xlabel('$R~/~\mathrm{AU}$',fontsize=13)

                # Mass loss on the right axes
                rha = lha.twinx()
                rha.loglog(grid.Rc, M_evap * (2 * np.pi), label='Raw Mass Lost ($M_{400}(\Sigma)$)', color=cycle[1])
                rha.loglog(grid.Rc, M_dot * (2 * np.pi), label='Weighted Mass Lost ($M_{400}(\Sigma)$)', color=cycle[2])
                rha.set_ylabel('$\dot{M}/~\mathrm{g~yr}^{-1}$',fontsize=13)

                if (mass_loss_mode == 'compare'):
                    photoevapM = photoevaporation.FRIEDExternalEvaporationM(model.disc)
                    (M_evapM, _) = photoevapM.unweighted_rates(model.disc)
                    rha.loglog(grid.Rc, M_evapM * (2 * np.pi), label='Mass Lost ($M_{integ}$)', color=cycle[3],marker='x')

                    photoevapS = photoevaporation.FRIEDExternalEvaporationS(model.disc)
                    (M_evapS, _) = photoevapS.unweighted_rates(model.disc)
                    rha.loglog(grid.Rc, M_evapS * (2 * np.pi), label='Mass Lost ($\Sigma$)', color=cycle[4])

                if (i==0):
                    ylimsr = [2e-11* Msun,5*np.max(M_evap*2*np.pi)]
                rha.set_ylim(ylimsr)

                plt.xlim([1,500])
                plt.legend(loc=2)

            #### SUBPLOT 3 ###
            # Dust size distributions
            if (isinstance(model.disc,DustGrowthTwoPop)):
                plt.subplot(no_plots,1,no_plots-1)
                
                # Large/small boundary
                St_eq = model.disc._eos.alpha/2
                a_eq = 2/np.pi * St_eq * model.disc.Sigma_G/model.disc._rho_s
                plt.loglog(grid.Rc, a_eq, label='$a_{eq}$')
                
                # Maximum entrained size
                if (model.photoevap is not None):
                    a_ent = model.photoevap._amax
                    plt.loglog(grid.Rc, a_ent, label='$a_{ent}$')
                else:
                    a_ent = np.zeros_like(a_eq)

                # Maximum size of dust
                a_max = model.disc.grain_size[1,:].flatten()
                plt.loglog(grid.Rc, a_max, label='$a_{max}$')

                plt.ylabel('$a~/~\mathrm{cm}$',fontsize=13)
                plt.xlim([1,500])
                plt.ylim( [ model.disc._monomer/10.0 , 10.0*np.amax(np.maximum.reduce([a_max,a_eq,a_ent])) ] )
                plt.legend(loc=6)

            ### SUBPLOT 4 ###
            # Dust mass fluxes
            """
            if (model.dust != None):
                plt.subplot(no_plots,1,no_plots)
                # Calculate drift velocities
                drift_velocities = model.dust._compute_deltaV(model.disc)
                # Average radius of pairs of cells
                R_av = 0.5 * ( grid.Rc[:-1] + grid.Rc[1:] )
                not_empty = (model.disc.Sigma[:-1] > 1e-5)
                fluxes = 0
                for eps_k, dV_k, a_k, St_k in zip(model.disc.dust_frac, drift_velocities, model.disc.grain_size, model.disc.Stokes()):
                    eps_a = a_k * eps_k
                    fluxes += model.dust._fluxes(model.disc, eps_a, dV_k, St_k)
                plt.loglog(grid.Rc,-fluxes,label='Large')
                plt.ylim([1e-9,1e-3])                
                plt.xlim([1,500])
                plt.ylabel('Dust Mass Flux')
                plt.legend()
            """

            ### SUBPLOT 4 ###
            # Dust-to-gas ratio
            if (isinstance(model.disc,DustGrowthTwoPop)):
                plt.subplot(no_plots,1,no_plots)
                plt.loglog(grid.Rc, model.disc.dust_frac.sum(0))
                plt.ylim([1e-4,1])                
                plt.xlim([1,500])
                plt.ylabel('Dust-to-gas ratio',fontsize=13)
                plt.legend()

            ## Measure disc properties and record
            R_outer = model.disc.Rout() # Locate outer radius by the density threshold
            if (model.photoevap is not None):
                R_outer = model.disc.Rot(model.photoevap) # If photoevaporation active, replace outer radius by position of mass loss rate maximum
            M_tot = model.disc.Mtot() # Total disc mass
            if (isinstance(model.disc,DustGrowthTwoPop)):
                model.disc.Mdust() # Remaining dust mass
                model.disc.Rdust_new(dust_radii_thresholds) # Radius containing proportion of dust mass
                model.disc._Mwind=np.append(model.disc._Mwind,[model.disc._Mwind_cum]) # Total mass of dust lost in wind.

            # Track accretion rate
            M_visc_out = 2*np.pi * grid.Rc[0] * model.disc.Sigma_G[0] * model._gas.viscous_velocity(model.disc)[0] * (AU**2)
            model.disc._Mdot_acc = np.append(model.disc._Mdot_acc,[-M_visc_out*(yr/Msun)])
            # Photoevaporative mass loss recorded through Track=True when the weighted rate is plotted

            # Finish plot (xlabel, title)
            plt.xlabel('$R~/~\mathrm{AU}$',fontsize=13)
            if (model.t>0):    
                logt = np.log10(model.t / (2 * np.pi))
                exponent = int(logt)
                prefactor = np.power(10,logt-exponent)
                plt.suptitle('Radial Profiles at $t={:.2f}\\times10^{:d}~\mathrm{{yr}}$ ($M_{{disc}}={:.3f}~M_J$)'.format(prefactor,exponent,(M_tot/Mjup)), fontsize=16)
            else:
                plt.suptitle('Radial Profiles at $t=0~\mathrm{{yr}}$ ($M_{{disc}}={:.3f}~M_J$)'.format((M_tot/Mjup)), fontsize=16)
            plt.tight_layout()
            plt.gcf().subplots_adjust(top=1-0.16/no_plots)
            # Save plot
            if model._gas:
                plt.savefig(plot_name+"_profiles/"+plot_name+"_{}.png".format(i))
            else:
                plt.savefig(plot_name+"_profiles/"+plot_name+"_novisc_{}.png".format(i))
            plt.close()

            # Save state
            save_summary(model,all_in)

            np.seterr(**err_state)

        io.pop_events(model.t)

def timeplot(model, plotting_data, nv_data, data_2=None,logtest=False):
    # Extract plot name
    plot_name = model['output']['plot_name']
    # Calculate the viscous timescale using the x=0 T=1 limit of the eqn in Clarke 2007
    M_d0 = model['disc']['mass'] / (1-np.exp(-model['grid']['R1']/model['disc']['Rc']))
    t_visc = M_d0*Mjup/Msun / (2 * plotting_data[:,5][0])

    # Work out how many plots needed
    no_data = np.shape(plotting_data)[1]
    if (no_data == 6):
        no_plots = 3
    else:
        no_plots = 6
    no_cols = int(no_plots/3)
    
    # Initiate the figure
    plt.subplots(3,no_cols,sharex=True,figsize=(10,12))
    plt.gcf().subplots_adjust(hspace=0,wspace=0,top=0.94,bottom=0.06)
    plt.rcParams['text.usetex'] = "True"
    plt.rcParams['font.family'] = "serif"

    def add_ref_t(ax,t):
        xlims = ax.get_xlim()
        ylims = ax.get_ylim()
        ax.semilogx([t,t],plt.ylim(), linestyle='-.', color='purple')
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)

    ### SUBPLOT 1 - Gas Radius
    plt.subplot(3,no_cols,1)
    if (np.max(plotting_data[:,2])>0):
        plt.semilogx(plotting_data[1:,0], plotting_data[1:,2], marker='x',color='blue',linestyle='None',label='$R(\dot{M}_{max})$')
    else:
        plt.semilogx(plotting_data[1:,0], plotting_data[1:,1], marker='x',color='blue',linestyle='None',label='$R(\Sigma=10^{-5})$')
    if (isinstance(nv_data,np.ndarray)) :
        xlims=plt.xlim()
        ylims=plt.ylim()   
        plt.semilogx(nv_data[1:,0], nv_data[1:,1], color='black',linestyle='--',label='Expected from initial timescale')
        plt.xlim(xlims)
        plt.ylim(ylims)
        plt.legend()
    add_ref_t(plt.gca(),t_visc)
    plt.ylabel('Disc Radius / AU',fontsize=16)
    plt.tick_params(axis='x', which='major', top=False, bottom=True, direction='inout', size=7)
    plt.tick_params(axis='x', which='minor', top=False, bottom=True, direction='inout', size=4)
    plt.tick_params(axis='both', which='major', labelsize=12)

    ### SUBPLOT 2 - Gas Mass
    plt.subplot(3,no_cols,1+no_cols)
    plt.semilogx(plotting_data[1:,0], plotting_data[1:,3]/Mjup, marker='x',color='blue',linestyle='None',label='Total')
    add_ref_t(plt.gca(),t_visc)
    plt.ylabel('Total Mass / $M_J$',fontsize=16)
    plt.tick_params(axis='x', which='major', top=True, bottom=True, direction='inout', size=7)
    plt.tick_params(axis='x', which='minor', top=True, bottom=True, direction='inout', size=4)
    plt.tick_params(axis='both', which='major', labelsize=12)

    ### SUBPLOT 3 - Mass Loss Rates
    plt.subplot(3,no_cols,1+2*no_cols)
    plt.loglog(plotting_data[:,0],plotting_data[:,4],label='$\dot{M}_{evap}$',linestyle='--',color='black')
    plt.loglog(plotting_data[:,0],plotting_data[:,5],label='$\dot{M}_{acc}$',linestyle='-',color='black')
    add_ref_t(plt.gca(),t_visc)
    plt.legend()
    plt.xlabel('Time / years',fontsize=16)
    plt.ylabel('$\dot{M} /M_\odot\mathrm{yr}^{-1}$',fontsize=16)
    plt.tick_params(axis='x', which='major', top=True, bottom=True, direction='inout', size=7)
    plt.tick_params(axis='x', which='minor', top=True, bottom=True, direction='inout', size=4)
    plt.tick_params(axis='both', which='major', labelsize=12)

    # Dust plots
    if (no_cols>1):
       ### SUBPLOT 1 - Dust Radii
       ax = plt.subplot(3,no_cols,2)
       cmap = plt.get_cmap("autumn")
       i_range = np.arange(0,np.shape(plotting_data)[1]-8,1)
       ax.set_prop_cycle('color', cmap(i_range/(i_range[-1]+1)))
       for i in i_range:
           ax.semilogx(plotting_data[1:,0], plotting_data[1:,i+8], marker='x',linestyle='None',label='{}\%'.format(100*np.array(model['dust']['radii_thresholds'])[i]))
       plt.legend()
       plt.ylabel('Dust Radius / AU',fontsize=16)
       ax.yaxis.tick_right()
       ax.yaxis.set_label_position("right")
       plt.tick_params(axis='x', which='major', top=False, bottom=True, direction='inout', size=7)
       plt.tick_params(axis='x', which='minor', top=False, bottom=True, direction='inout', size=4)
       plt.tick_params(axis='both', which='major', labelsize=12)

       ### SUBPLOT 2 - Dust Masses
       ax = plt.subplot(3,no_cols,2+no_cols)
       plt.semilogx(plotting_data[1:,0], plotting_data[1:,6]/Mjup, marker='x',color='blue',linestyle='None',label='Mass left in Disc')
       plt.semilogx(plotting_data[1:,0], plotting_data[1:,7]/Mjup, marker='x',color='red',linestyle='None',label='Mass lost in Wind')
       accretion_loss = (plotting_data[0,6]-plotting_data[:,6])/Mjup - plotting_data[:,7]/Mjup
       plt.semilogx(plotting_data[1:,0], accretion_loss[1:], marker='x',color='green',linestyle='None',label='Mass lost in Accretion/Drift')
       plt.legend()
       plt.xlabel('Time / years',fontsize=16)
       plt.ylabel('Dust Mass $M_d$ / $M_J$',fontsize=16)
       ax.yaxis.tick_right()
       ax.yaxis.set_label_position("right")
       plt.tick_params(axis='x', which='major', top=True, bottom=True, direction='inout', size=7)
       plt.tick_params(axis='x', which='minor', top=True, bottom=True, direction='inout', size=4)
       plt.tick_params(axis='both', which='major', labelsize=12)

       def logistic(t, t0, k):
           return 1 / (1 + (t/t0)**(-k))

       if logtest:
           norm_acc = accretion_loss/accretion_loss[-1]
           popt, pcov = scp.optimize.curve_fit(logistic, plotting_data[norm_acc<0.5,0], norm_acc[norm_acc<0.5], p0=[1e5,1.5])
           plt.semilogx(plotting_data[1:,0], accretion_loss[-1]*logistic(plotting_data[1:,0], *popt), linestyle='--')
           print(popt)

       print ("Fraction lost to wind: {:.3f}".format(plotting_data[-1,7] / plotting_data[0,6]))
       print ("Fraction lost to star: {:.3f}".format(accretion_loss[-1]*Mjup / plotting_data[0,6]))

       ### SUBPLOT 3 - Either dust loss rates or residuals of mass loss
       ax = plt.subplot(3,no_cols,2+2*no_cols)
       if (data_2 is not None):
           accretion_loss_2 = (data_2[1:,7][0]-data_2[1:,7] - data_2[1:,8])/Mjup
           print(accretion_loss_2)
           ax.semilogx(plotting_data[1:,0], accretion_loss - accretion_loss_2, marker='x',color='blue',linestyle='None')
           plt.ylabel('Relative accretion mass loss $\Delta M_{d,acc}$ / $M_J$')
           ax.yaxis.tick_right()
           ax.yaxis.set_label_position("right")
       else:
           ### SUBPLOT 3 - Rate of dust loss and derivative
           wind_rate = plotting_data[:,0]*np.gradient(plotting_data[:,7]/Mjup, plotting_data[:,0])
           drift_rate = plotting_data[:,0]*np.gradient(accretion_loss, plotting_data[:,0])
           plt.semilogx(plotting_data[1:,0],wind_rate[1:], color='red')
           plt.semilogx(plotting_data[1:,0],drift_rate[1:], color='green')
           plt.ylabel('Dust Mass Loss Rate $\\frac{dM_d}{d\log(t)}$ / $M_J$',fontsize=16)

           total_rate = wind_rate + drift_rate
           second_deriv = plotting_data[:,0]*np.gradient(total_rate, plotting_data[:,0])
           drain_t = plotting_data[:,0][np.argmin(second_deriv)]
           print ("Dust lost after: {:.3f} Myr".format(drain_t/1e6))
           #plt.semilogx(plotting_data[1:,0],-second_deriv[1:])
           #plt.ylabel('Dust Loss Rate $\\frac{d^2M_d}{d\log(t)^2}$ / $M_J$',fontsize=16)

           plt.xlabel('Time / years',fontsize=16)
           ax.yaxis.tick_right()
           ax.yaxis.set_label_position("right")
           plt.tick_params(axis='x', which='major', top=True, bottom=True, direction='inout', size=7)
           plt.tick_params(axis='x', which='minor', top=True, bottom=True, direction='inout', size=4)
           plt.tick_params(axis='both', which='major', labelsize=12)

    # Save Figure
    plt.suptitle('Evolution of the disc',fontsize='24')
    if logtest:
        plt.savefig(model['output']['plot_name']+"_logtest.png")
        plt.show()
    elif (nv_data is not None):
        plt.savefig(model['output']['plot_name']+"_time.png")
    plt.close()

def setup_wrapper(model,):
    # Setup model
    disc = setup_disc(model)
    driver = setup_model(model, disc)
    output_name, io_control = setup_output(model)
    plot_name = model['output']['plot_name']

    # Get plotting limits based on original input
    plt.figure()
    plt.loglog(disc.R,disc.Sigma_G)
    if (isinstance(disc,DustGrowthTwoPop)):
        plt.loglog(disc.R,disc.Sigma_D.sum(0))
    ylims = plt.ylim()
    ylims = (ylims[0]*1e-3, ylims[1])
    plt.close()

    # Truncate disc at base of wind
    if (driver.photoevap is not None):
        optically_thin = (disc.R > disc.Rot(driver.photoevap))

        """Could consider more whether dust left behind or drifts in
           - under strong mass loss likelyto mainly be entrained and lost at monomer size
           - at weaker mass loss may be left behind - but I don't like how it behaves at the moment so assuming it isn't"""
        disc._Sigma[optically_thin] = 0
        """disc._Sigma[optically_thin] = disc.Sigma_D.sum(0)[optically_thin]
        f_ratio = disc.dust_frac[1,optically_thin] / disc.dust_frac[1,optically_thin]
        disc.dust_frac[1,optically_thin] = 0.99999*f_ratio*np.ones_like(disc.dust_frac[1,optically_thin])
        disc.dust_frac[0,optically_thin] = 0.99999*(1-f_ratio)*np.ones_like(disc.dust_frac[0,optically_thin])"""
    """Lines to truncate with no mass loss if required for direct comparison"""
    """else:
        photoevap = photoevaporation.FRIEDExternalEvaporationMS(disc)
        optically_thin = (disc.R > disc.Rot(photoevap))"""
    
    Dt_nv = np.zeros_like(disc.R)
    if (driver.photoevap is not None):
        # Perform estimate of evolution for non-viscous case
        (_, _, M_cum, Dt_nv) = driver.photoevap.get_timescale(disc)

    return disc, driver, output_name, io_control, plot_name, ylims, Dt_nv

def main():
    # Retrieve model from inputs
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default=DefaultModel)
    args = parser.parse_args()
    model = json.load(open(args.model, 'r'))
    
    # Do all setup
    disc, driver, output_name, io_control, plot_name, ylims, Dt_nv = setup_wrapper(model)
    
    # Run model
    run(driver, io_control, output_name, plot_name, ylims, model['uv']['photoevaporation'], np.array(model['dust']['radii_thresholds']), model)
    
    # Save disc properties
    outputdata = save_summary(driver,model)

    # Call separate plotting function
    timeplot(model, outputdata, np.column_stack((Dt_nv/(2*np.pi), disc.grid.Rc)))

    # Compile video
    subprocess.call(['ffmpeg', '-framerate', '5', '-i', model['output']['plot_name']+'_profiles/'+model['output']['plot_name']+'_%01d.png', model['output']['plot_name']+'.avi'])

def save_summary(driver,model):
    # Retrieve disc properties
    outer_radii = driver.disc._Rout
    ot_radii = driver.disc._Rot
    disc_masses = driver.disc._Mtot
    if (driver.photoevap is not None):
        Mevap = driver.photoevap._Mdot
    Macc = driver.disc._Mdot_acc
    if (isinstance(driver.disc,DustGrowthTwoPop)):
        dust_radii = driver.disc._Rdust
        dust_masses = driver.disc._Mdust
        if (driver.photoevap is not None):
            dust_wind = driver.disc._Mwind
        dust_split = np.split(dust_radii,range(1,np.shape(dust_radii)[1]),axis=1)

    # Save data
    plot_times = np.array(model['output']['plot_times'])
    if (driver.photoevap is None and isinstance(driver.disc,DustGrowthTwoPop)):
        outputdata = np.column_stack((plot_times[0:np.size(outer_radii):1],outer_radii,np.zeros_like(outer_radii),disc_masses,np.zeros_like(outer_radii),Macc,dust_masses,np.zeros_like(outer_radii),*dust_split))
    elif (driver.photoevap is None):
        outputdata = np.column_stack((plot_times[0:np.size(outer_radii):1],outer_radii,np.zeros_like(outer_radii),disc_masses,np.zeros_like(outer_radii),Macc))
    elif (isinstance(driver.disc,DustGrowthTwoPop)):
        outputdata = np.column_stack((plot_times[0:np.size(outer_radii):1],outer_radii,ot_radii[1:],disc_masses,Mevap,Macc,dust_masses,dust_wind,*dust_split))
    else:
        outputdata = np.column_stack((plot_times[0:np.size(outer_radii):1],outer_radii,ot_radii[1:],disc_masses,Mevap,Macc))
    np.savetxt(model['output']['directory']+"/"+model['output']['plot_name']+"_discproperties.dat",outputdata)

    return outputdata

if __name__ == "__main__":
    main()
    
