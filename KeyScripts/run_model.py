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
    Sigma *= Msun / AU**2

    eos.update(0, Sigma)

    try:
        feedback = model['disc']['feedback']
    except KeyError:
        feedback = True

    # If non-zero dust, set up a two population model, else use a simple accretion disc
    if (model['disc']['d2g']>0):
        disc = DustGrowthTwoPop(grid, star, eos, p['d2g'], Sigma=Sigma, Sc=model['disc']['Schmidt'], feedback=feedback)
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
    if (model['uv']['photoevaporation'] == "FRIED"):
        photoevap = photoevaporation.FRIEDExternalEvaporationMS(disc) # Using 2DMS at 400
    elif (model['uv']['photoevaporation'] == "Constant"):
        photoevap = photoevaporation.FixedExternalEvaporation(1e-9)
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
    output_times = np.arange(out['first'], out['last'], out['interval'])
    if not np.allclose(out['last'], output_times[-1], 1e-12):
        output_times = np.append(output_times, out['last'])

    output_times *= yr
    
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
    
    
def run(model, io, base_name, plot_name, ylims, mass_loss_mode, verbose=True, n_print=1000):
    i=0
    threshold = model.disc.Sigma[-1]
    while not io.finished():
        ti = io.next_event_time()
        while model.t < ti:
            """The model breaks down when all at base rate because i_max = i_edge, Sigma(i_edge -> 0). Instead, terminate the model"""
            not_empty = (model.disc.Sigma_G > 0)
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
           
            (M_evap, M_ann) = model.photoevap.unweighted_rates(model.disc)

            plt.figure()
            plt.rcParams['text.usetex'] = "True"
            i+=1
            no_plots = 2 + int(model.photoevap == None) + int(isinstance(model.disc,DustGrowthTwoPop))
            
            # Plot of density profile
            plt.subplot(no_plots,1,1)
            plt.loglog(grid.Rc, model.disc.Sigma_G, label='$\Sigma_\mathrm{G}$')
            if (isinstance(model.disc,DustGrowthTwoPop)):
                plt.loglog(grid.Rc, model.disc.Sigma_D.sum(0), label='$\Sigma_\mathrm{D}$', linestyle='--')
            plt.ylabel('$\Sigma$')
            plt.xlim([1,500])
            plt.ylim(ylims)
            plt.legend(loc=3)

            # Plot of weighted mass loss rate profiles and comparisons
            plt.subplot(no_plots,1,2)
            cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

            if (model.photoevap == None):
                """(M_dot_MS,M_ann) = photoevap.optically_thin_weighting(model.disc)
                (M_dot_S,_) = photoevap2.optically_thin_weighting(model.disc)
                plt.loglog(grid.Rc, M_ann, label='Annulus Mass')
                plt.loglog(grid.Rc, M_evapMS*dt, label='Raw Mass Lost $(M)$')
                plt.loglog(grid.Rc, M_evapS*dt, label='Raw Mass Lost $(\Sigma)$')
                plt.loglog(grid.Rc, M_dot_MS*dt, label='Weighted Mass Lost $(M)$')
                plt.loglog(grid.Rc, M_dot_S*dt, label='Weighted Mass Lost $(\Sigma)$')"""

            lha = plt.gca()
            (M_dot, _) = model.photoevap.optically_thin_weighting(model.disc, Track=True)
            lha.loglog(grid.Rc, M_ann, label='Annulus Mass')
            lha.set_ylabel('$M/~\mathrm{g}$')
            rha = lha.twinx()

            if (mass_loss_mode == 'compare'):
                photoevapM = photoevaporation.FRIEDExternalEvaporationM(model.disc)
                (M_evapM, _) = photoevapM.unweighted_rates(model.disc)
                rha.loglog(grid.Rc, M_evapM * (2 * np.pi), label='Mass Lost ($M_{integ}$)', color=cycle[3],marker='x')

                photoevapS = photoevaporation.FRIEDExternalEvaporationS(model.disc)
                (M_evapS, _) = photoevapS.unweighted_rates(model.disc)
                rha.loglog(grid.Rc, M_evapS * (2 * np.pi), label='Mass Lost ($\Sigma$)', color=cycle[4])

            rha.loglog(grid.Rc, M_evap * (2 * np.pi), label='Raw Mass Lost ($M_{400}(\Sigma)$)', color=cycle[1])
            rha.loglog(grid.Rc, M_dot * (2 * np.pi), label='Weighted Mass Lost ($M_{400}(\Sigma)$)', color=cycle[2])
            rha.set_ylabel('$\dot{M}/~\mathrm{g~yr}^{-1}$')

            if (i==1):
                ylims2 = [1e-12* Msun,10*np.max(M_evap)]
                yrange = ylims2[1]/ylims2[0]
                ylimsl = lha.get_ylim()
            rha.set_ylim([1e-12* Msun,yrange*1e-12* Msun])
            lha.set_ylim(ylimsl)

            plt.xlim([1,500])
            plt.xlabel('$R/~\mathrm{AU}$')
            plt.legend(loc=2)

            if (model.photoevap == None):
                # Plot of actual mass loss rate profiles and comparisons 
                plt.subplot(313)
                plt.semilogy(grid.Rc, M_ann / M_ann[-1], label='Annulus Mass')
                plt.semilogy(grid.Rc, M_evapMS / M_evapMS[-1], label='Mass Lost $(M)$')
                plt.semilogy(grid.Rc, M_evapS / M_evapS[-1], label='Mass Lost $(\Sigma)$')
                RSigma = grid.Rc * model.disc.Sigma_G
                RSigma *= M_evap[-1] / RSigma[-1]
                plt.semilogy(grid.Rc, RSigma / RSigma[-1], label='$R\Sigma$ Limit')
                plt.semilogy(grid.Rc, np.ones_like(grid.Rc), linestyle='--', color='black')
                plt.semilogy(grid.Rc, np.power(grid.Rc/grid.Rc[-1],1.5)*model.disc.Sigma_G/model.disc.Sigma_G[-1], linestyle='--', color='blue', label='$R^{3/2} \Sigma$')

                plt.xlabel('$R/\mathrm{AU}$')
                plt.ylabel('$M$ (normalised)')
                plt.xlim([100,400])
                ylims3 = plt.ylim()
                plt.ylim([1,ylims3[1]])
                plt.legend()

            if (isinstance(model.disc,DustGrowthTwoPop)):
                plt.subplot(313)
                
                St_eq = model.disc._eos.alpha/2
                a_eq = 2/np.pi * St_eq * model.disc.Sigma_G/model.disc._rho_s
                plt.loglog(grid.Rc, a_eq, label='$a_{eq}$')
                
                #a_ent = model.photoevap.max_size_entrained(model.disc)
                a_ent = model.photoevap._amax
                plt.loglog(grid.Rc, a_ent, label='$a_{ent}$')

                a_max = model.disc.grain_size[1,:].flatten()
                plt.loglog(grid.Rc, a_max, label='$a_{max}$')

                plt.xlabel('$R/\mathrm{AU}$')
                plt.ylabel('$a$')
                plt.xlim([1,500])
                ylims3 = plt.ylim()
                plt.ylim([model.disc._amin/10.0,10.0*np.amax(np.maximum.reduce([a_max,a_eq,a_ent]))])
                plt.legend(loc=6)

            # Locate outer radius
            R_outer = model.disc.Rot(model.photoevap) #Rot = optically thin, Rout=10^-5
            R_outer = model.disc.Rout()
            M_tot = model.disc.Mtot()
            if (isinstance(model.disc,DustGrowthTwoPop)):
                model.disc.Mdust()
                model.disc.Rdust()

            # Track accretion rate
            M_visc_out = 2*np.pi * grid.Rc[0] * model.disc.Sigma_G[0] * model._gas.viscous_velocity(model.disc)[0] * (AU**2)
            model.disc._Mdot_acc = np.append(model.disc._Mdot_acc,[-M_visc_out*(yr/Msun)])

            # Add title and save plot
            if (model.t>0):    
                logt = np.log10(model.t / (2 * np.pi))
                exponent = int(logt)
                prefactor = np.power(10,logt-exponent)
                plt.suptitle('Surface density and mass loss rates at $t={:.2f}\\times10^{:d}~\mathrm{{yr}}$ ($M_{{disc}}={:.3f}~M_\odot$, $R_{{disc}}={:4.1f})$'.format(prefactor,exponent,(M_tot/Msun),R_outer))
            else:
                plt.suptitle('Surface density and mass loss rates at $t=0~\mathrm{{yr}}$ ($M_{{disc}}={:.3f}~M_\odot$, $R_{{disc}}={:4.1f})$'.format((M_tot/Msun),R_outer))
            if model._gas:
                plt.savefig(plot_name+"_profiles/"+plot_name+"_{}.png".format(i))
            else:
                plt.savefig(plot_name+"_profiles/"+plot_name+"_novisc_{}.png".format(i))
            plt.close()

            #print (model.disc.tot_mass_lost/Msun)

            np.seterr(**err_state)

        io.pop_events(model.t)

def timeplot(model, radius_data, nv_data, t_visc=None):
    plot_name = model['output']['plot_name']

    no_plots = np.shape(radius_data)[1]-3
    no_cols = int((no_plots-1)/2)
    
    plt.subplots(3,no_cols,sharex=True,figsize=(10,12))
    plt.gcf().subplots_adjust(hspace=0,wspace=0,top=0.92)
    plt.rcParams['text.usetex'] = "True"

    plt.subplot(3,no_cols,1)

    plt.semilogx(radius_data[:,0], radius_data[:,1], marker='x',color='blue',linestyle='None',label='$R(\Sigma=10^{-5})$')
    plt.semilogx(radius_data[:,0], radius_data[:,2], marker='x',color='red',linestyle='None',label='$R(\dot{M}_{max})$')
    if (isinstance(nv_data,np.ndarray)) :
        xlims=plt.xlim()
        ylims=plt.ylim()
        plt.semilogx(nv_data[:,0], nv_data[:,1], color='black',linestyle='--',label='Expected from initial timescale')
        plt.xlim(xlims)
        plt.ylim(ylims)
    plt.legend()
    xlims = plt.xlim()
    plt.semilogx([t_visc,t_visc],plt.ylim(), linestyle='-.', color='purple')
    plt.xlim(xlims)
    plt.ylabel('Disc Radius / AU')

    plt.subplot(3,no_cols,1+no_cols)
    plt.semilogx(radius_data[:,0], radius_data[:,3]/Mjup, marker='x',color='blue',linestyle='None',label='Total')
    xlims = plt.xlim()
    plt.semilogx([t_visc,t_visc],plt.ylim(), linestyle='-.', color='purple')
    plt.xlim(xlims)
    plt.ylabel('Total Mass / $M_J$')

    plt.subplot(3,no_cols,2+no_cols)
    plt.loglog(radius_data[:,0],radius_data[:,4],label='$\dot{M}_{evap}$',linestyle='--',color='black')
    plt.loglog(radius_data[:,0],radius_data[:,5],label='$\dot{M}_{acc}$',linestyle='-',color='black')
    xlims = plt.xlim()
    plt.semilogx([t_visc,t_visc],plt.ylim(), linestyle='-.', color='purple')
    plt.xlim(xlims)
    plt.legend()
    plt.xlabel('Time / years')
    plt.ylabel('$\dot{M} /M_\odot\mathrm{yr}^{-1}$')

    if (no_cols>1):
       ax = plt.subplot(3,no_cols,2)
       plt.semilogx(radius_data[:,0], radius_data[:,6], marker='x',color='blue',linestyle='None')
       plt.ylabel('Dust Radius / AU')
       ax.yaxis.tick_right()
       ax.yaxis.set_label_position("right")

       ax = plt.subplot(3,no_cols,2+no_cols)
       plt.semilogx(radius_data[:,0], radius_data[:,7]/Mjup, marker='x',color='blue',linestyle='None',label='Dust')
       plt.xlabel('Time / years')
       plt.ylabel('Dust Mass / $M_J$')	
       ax.yaxis.tick_right()
       ax.yaxis.set_label_position("right")

    plt.suptitle('Evolution of the disc',fontsize='20')
    plt.savefig(model['output']['plot_name']+"_time.png")
    plt.close()

def main():
    # Retrieve model from inputs
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default=DefaultModel)
    args = parser.parse_args()
    model = json.load(open(args.model, 'r'))
    
    # Setup model
    disc = setup_disc(model)
    driver = setup_model(model, disc)
    output_name, io_control = setup_output(model)
    plot_name = model['output']['plot_name']

    # Truncate disc at base of wind
    plt.figure()
    plt.loglog(disc.R,disc.Sigma_G)
    if (isinstance(disc,DustGrowthTwoPop)):
        plt.loglog(disc.R,disc.Sigma_D.sum(0))
    ylims = plt.ylim()
    ylims = (ylims[0]*1e-3, ylims[1])
    plt.close()
    optically_thin = (disc.R > disc.Rot(driver.photoevap))
    disc._Sigma[optically_thin] *= 0.0 # Turned off for interpolation testing
    if (isinstance(disc,DustGrowthTwoPop)):
        disc.Rdust()
    M_visc_out = 2*np.pi * disc.grid.Rc[0] * disc.Sigma_G[0] * driver._gas.viscous_velocity(disc)[0] * (AU**2)
    driver.photoevap.optically_thin_weighting(disc, Track=True)
    disc._Mdot_acc = np.append(disc._Mdot_acc,[-M_visc_out*(yr/Msun)])
    
    M_d0 = model['disc']['mass'] / (1-np.exp(-model['grid']['R1']/model['disc']['Rc']))
    t_visc = M_d0 / (-2 * M_visc_out*(yr/Msun))

    """Could consider more whether dust left behind or drifts in - likely to mainly be entrained and lost due to size"""

    # Perform estimate of evolution for non-viscous case
    (_, _, M_cum, Dt_nv) = driver.photoevap.get_timescale(disc)
    
    # Run model and retrieve disc properties
    run(driver, io_control, output_name, plot_name, ylims, model['uv']['photoevaporation'])
    outer_radii = driver.disc._Rout
    ot_radii = driver.disc._Rot
    disc_masses = driver.disc._Mtot
    Mevap = driver.photoevap._Mdot
    Macc = driver.disc._Mdot_acc
    if (isinstance(disc,DustGrowthTwoPop)):
        dust_radii = driver.disc._Rdust
        dust_masses = driver.disc._Mdust

    # Save radius data
    #plot_times=np.insert(np.array(model['output']['plot_times']),0,0)
    plot_times = np.array(model['output']['plot_times'])
    if (isinstance(disc,DustGrowthTwoPop)):
        outputdata = np.column_stack((plot_times[0:np.size(outer_radii)-1:1],outer_radii[1:],ot_radii[1:],disc_masses[1:],Mevap[1:],Macc[1:],dust_radii[1:],dust_masses[1:]))
    else:
        outputdata = np.column_stack((plot_times[0:np.size(outer_radii)-1:1],outer_radii[1:],ot_radii[1:],disc_masses[1:],Mevap[1:],Macc[1:]))
    np.savetxt(model['output']['directory']+"/"+model['output']['plot_name']+"_discproperties.dat",outputdata)

    # Call separate plotting function
    timeplot(model, outputdata[1:,:], np.column_stack((Dt_nv/(2*np.pi), disc.grid.Rc)), t_visc) 

if __name__ == "__main__":
    main() 


    




