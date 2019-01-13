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
from DiscEvolution.constants import Msun, AU, yr
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
        photoevap = photoevaporation.FRIEDExternalEvaporationS(disc) # Using 3DS
    elif (model['uv']['photoevaporation'] == "Constant"):
        photoevap = photoevaporation.FixedExternalEvaporation(1e-9)
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
    base_name = os.path.join(out['directory'], out['base'] + '_{:04d}')

    format = out['format']
    if format.lower() == 'hdf5':
        base_name += '.h5'
    elif format.lower() == 'ascii':
        base_name += '.dat'
    else:
        raise ValueError ("Output format {} not recognized".format(format))

    return base_name, EC
    
    
def run(model, io, base_name, plot_name, outer_radii, verbose=True, n_print=1000):
    i=0
    threshold = model.disc.Sigma_G[-1]
    while not io.finished():
        ti = io.next_event_time()
        while model.t < ti:
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
           
            if (model.photoevap == None):
                photoevapS = photoevaporation.FRIEDExternalEvaporationS(model.disc)
                (M_evapS, M_ann, M_cum, Dt) = photoevap.get_timescale(model.disc,dt)
                photoevapMS = photoevaporation.FRIEDExternalEvaporationMS(model.disc)
                (M_evapMS, _, _, _) = photoevap2.get_timescale(model.disc,dt)
            else:
                (M_evap, M_ann, M_cum, Dt) = model.photoevap.get_timescale(model.disc,dt)

            plt.figure()
            plt.rcParams['text.usetex'] = "True"
            i+=1
            no_plots = 2+ int(model.photoevap == None)
            
            # Plot of density profile
            plt.subplot(no_plots,1,1)
            plt.loglog(grid.Rc, model.disc.Sigma_G, label='$\Sigma_\mathrm{G}$')
            #plt.xlabel('$R/\mathrm{AU}$')
            plt.ylabel('$\Sigma$')
            plt.xlim([1,500])
            if (i==1):
                ylims = plt.ylim()
            else:
                plt.ylim(ylims)
            #plt.ylim([max(1e-15,ylims[0]),min(10*np.max(model.disc.Sigma_G),ylims[1])])
            plt.legend(loc=3)

            # Plot of weighted mass loss rate profiles and comparisons
            plt.subplot(no_plots,1,2)
            plt.loglog(grid.Rc, M_ann, label='Annulus Mass')
            if (model.photoevap == None):
                (M_dot_MS,_) = photoevap.optically_thin_weighting(model.disc,dt)
                (M_dot_S,_) = photoevap2.optically_thin_weighting(model.disc,dt)
                plt.loglog(grid.Rc, M_evapMS*dt, label='Raw Mass Lost $(M)$')
                plt.loglog(grid.Rc, M_evapS*dt, label='Raw Mass Lost $(\Sigma)$')
                plt.loglog(grid.Rc, M_dot_MS*dt, label='Weighted Mass Lost $(M)$')
                plt.loglog(grid.Rc, M_dot_S*dt, label='Weighted Mass Lost $(\Sigma)$')
            else:
                (M_dot,_) = model.photoevap.optically_thin_weighting(model.disc,dt)
                plt.loglog(grid.Rc, M_evap*dt, label='Raw Mass Lost')
                plt.loglog(grid.Rc, M_dot*dt, label='Weighted Mass Lost')

            plt.xlabel('$R/\mathrm{AU}$')
            plt.ylabel('$M$')
            plt.xlim([1,500])
            if (i==1):
                ylims2 = [1e-11* Msun / (2 * np.pi)*dt,10*np.max(M_ann)]
                yrange = ylims2[1]/ylims2[0]
            plt.ylim([1e-11* Msun / (2 * np.pi)*dt,yrange*1e-11* Msun / (2 * np.pi)*dt])
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

            # Locate outer radius (currently surface density greater than 10^{-10}
            # threshold = 1e-10
            notempty = model.disc.Sigma > threshold
            notempty_cells = grid.Rc[notempty]
            if np.size(notempty_cells>0):
                R_outer = notempty_cells[-1]
            else:
                R_outer = 0
            outer_radii = np.append(outer_radii,[R_outer])

            # Add title and save plot            
            logt = np.log10(model.t / (2 * np.pi))
            exponent = int(logt)
            prefactor = np.power(10,logt-exponent)
            plt.suptitle('Surface density and mass loss rates at $t={:.2f}\\times10^{:d}~\mathrm{{yr}}$ ($M_{{disc}}={:.3f}~M_\odot$, $R_{{disc}}={:4.1f})$'.format(prefactor,exponent,M_cum[0]/(Msun),R_outer))
            if model._gas:
                plt.savefig(plot_name+"_profiles/"+plot_name+"_visc_{}.png".format(i))
            else:
                plt.savefig(plot_name+"_profiles/"+plot_name+"_novisc_{}.png".format(i))
            plt.close()

            #print (model.disc.tot_mass_lost/Msun)

            np.seterr(**err_state)

        io.pop_events(model.t)

    return outer_radii

def timeplot(model, radius_data, nv_data):
    plot_name = model['output']['plot_name']
    
    radius_plot = plt.figure()
    plt.rcParams['text.usetex'] = "True"

    plt.semilogx(radius_data[:,0], radius_data[:,1], marker='x',color='blue',linestyle='None')
    xlims=plt.xlim()
    plt.semilogx(nv_data[:,0], nv_data[:,1], color='black',linestyle='--',label='Expected from initial timescale')
    plt.xlim(xlims)

    # Complete Plot
    plt.xlabel('Time / years')
    plt.ylabel('Radius of disc / AU')
    plt.title('Evolution of disc radius'.format(model['output']['interval']))
    plt.legend()
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

    """# Setup plot for radius against time
    time_plot = plt.figure()
    plt.rcParams['text.usetex'] = "True"""

    """position_Rc = np.argmin(np.abs(disc.grid.Rc-model['disc']['Rc']))
    print (disc.grid.Rc[position_Rc])
    viscous_t = model['disc']['Rc']**2 / (3*disc.nu[position_Rc])/(2*np.pi)
    print (viscous_t)"""

    # Perform estimate of evolution for non-viscous case
    (_, _, M_cum, Dt_nv) = driver.photoevap.get_timescale(disc,100)
    
    # Run model, returning outer radius of disc
    R_0 = model['grid']['R1']
    outer_radii = np.array(disc.R_edge[-1])
    outer_radii = run(driver, io_control, output_name, plot_name, outer_radii)

    """# (Fit and) Plot the trend for no viscosity
    plt.figure(time_plot.number)

    if (model['uv']['photoevaporation'] == "Constant"):
        t_vals = np.linspace(plt.xlim()[0],plt.xlim()[1],101)
        t_func = np.exp(-model['grid']['R1']/model['disc']['Rc']) + 1e-9 / ( model['star']['mass']*model['disc']['mass'] ) * (1-np.exp(-model['grid']['R1']/model['disc']['Rc'])) * t_vals
        R_vals = -1 * model['disc']['Rc'] * np.log(t_func)
        plt.plot(t_vals,R_vals,color='black',linestyle='--',label='Expected without viscous spreading')
    elif (model['uv']['photoevaporation'] == "FRIED"):
        t_sim = np.array([0])
        t_sim = np.append(t_sim,model['output']['plot_times'])
        def R_func(t, R_a, R_b, t0):
            R_vals = np.maximum(R_a * np.power((t+t0),-1.0/2.7), R_b)
            return R_vals
        (popt, pcov) = scp.optimize.curve_fit(R_func,t_sim[3:],outer_radii[3:],(4000,70,-1e3))
        plt.plot(t_vals[1:],R_func(t_vals[1:],*popt),color='black',linestyle='--',label='Expected without viscous spreading')
        print (popt)
    xlims=plt.xlim()
    plt.plot(Dt_nv/(2*np.pi), disc.grid.Rc, color='black',linestyle='--',label='Expected without viscous spreading')
    plt.xlim(xlims)
    ax=plt.gca()
    ax.set_xscale("log")
    if (ax.get_xscale()=="linear"):
        plt.plot(0, outer_radii[0],marker='x',color='blue',label='Viscous Simulation')

    # Complete Plot
    plt.xlabel('Time / years')
    plt.ylabel('Radius of disc / AU')
    plt.title('Evolution of disc radius'.format(model['output']['interval']))
    plt.legend()
    plt.savefig(model['output']['plot_name']+"_time.png")"""

    # Save radius data
    plot_times=np.insert(np.array(model['output']['plot_times']),0,0)
    outputdata = np.column_stack((plot_times,outer_radii))
    np.savetxt(plot_name+"_discproperties.dat",outputdata)

    # Check separate plotting function
    timeplot(model, outputdata[1:,:], np.column_stack((Dt_nv/(2*np.pi), disc.grid.Rc)))

if __name__ == "__main__":
    main() 


    




