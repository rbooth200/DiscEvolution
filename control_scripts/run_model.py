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
import matplotlib.pyplot as plt
from DiscEvolution.constants import Msun, AU, yr, Mjup
from DiscEvolution.grid import Grid
from DiscEvolution.star import SimpleStar, PhotoStar
from DiscEvolution.eos  import IrradiatedEOS, LocallyIsothermalEOS
from DiscEvolution.dust import DustGrowthTwoPop
from DiscEvolution.opacity import Tazzari2016
from DiscEvolution.viscous_evolution import ViscousEvolution, ViscousEvolutionFV
from DiscEvolution.disc import AccretionDisc
from DiscEvolution.dust import SingleFluidDrift
from DiscEvolution.diffusion import TracerDiffusion
from DiscEvolution.driver import DiscEvolutionDriver
from DiscEvolution.io import Event_Controller, DiscReader
from DiscEvolution.disc_utils import mkdir_p
from DiscEvolution.internal_photo import PrimordialDisc
import DiscEvolution.photoevaporation as photoevaporation
import FRIED.photorate as photorate
#import subprocess

###############################################################################
# Global Constants
###############################################################################

DefaultModel = "DiscConfig_default.json"

###############################################################################
# Setup Functions
###############################################################################

def LBP_profile(R,R_C,Sigma_C):
    # For profile fitting
    x = R/R_C
    return np.log(Sigma_C) - np.log(x)-x

def setup_disc(model):
    '''Create disc object from initial conditions'''
    # Setup the grid, star and equation of state
    p = model['grid']
    grid = Grid(p['R0'], p['R1'], p['N'], spacing=p['spacing'])

    p = model['star']
    try:
        if (model['x-ray']['L_X'] > 0):
            star = PhotoStar(LX=model['x-ray']['L_X'], M=model['star']['mass'], R=model['star']['radius'], T_eff=model['star']['T_eff'])
        else:
            star = SimpleStar(M=p['mass'], R=p['radius'], T_eff=p['T_eff'])
    except KeyError:
        star = SimpleStar(M=p['mass'], R=p['radius'], T_eff=p['T_eff'])
    
    p = model['eos']
    try:
        mu = p['mu']
    except KeyError:
        mu = 2.4
    if p['type'] == 'irradiated':
        assert p['opacity'] == 'Tazzari2016'
        kappa = Tazzari2016()
        eos = IrradiatedEOS(star, model['disc']['alpha'], kappa=kappa, mu=mu)
    elif p['type'] == 'iso':
        eos = LocallyIsothermalEOS(star, p['h0'], p['q'], 
                                   model['disc']['alpha'], mu=mu)
    else:
        raise ValueError("Error: eos::type not recognised")
    eos.set_grid(grid)

    # Setup the physical part of the disc
    p = model['disc']
    if (('profile' in model['disc']) == False):
        Sigma = np.exp(-grid.Rc / p['Rc']) / (grid.Rc) # Catch missing profile by assuming Lynden-Bell & Pringle
    elif (model['disc']['profile'] == 'LBP'):
        try:
            gamma_visc = model['disc']['gamma']
        except:
            gamma_visc = 1
        Sigma = np.exp(-(grid.Rc / p['Rc'])**(2-gamma_visc)) / (grid.Rc**gamma_visc) # Lynden-Bell & Pringle
    else:
        try:
            gamma_visc = model['disc']['gamma']
        except:
            gamma_visc = 1.5 + 2 * model['eos']['q']               # Set gamma to steady state
        Sigma = 1.0 / (grid.Rc**gamma_visc)                        # R^-gamma Power Law (default take match T(R))
    Sigma *= p['mass'] / np.trapz(Sigma, np.pi*grid.Rc**2)
    if (p['unit']=='jup'):          # Disc mass given in Jupiter masses
        Sigma *= Mjup / AU**2
    elif (p['unit']=='sol'):        # Disc mass given in Solar masses
        Sigma *= Msun / AU**2

    eos.update(0, Sigma)

    try:
        feedback = model['disc']['feedback']
    except KeyError:
        feedback = True

    # If non-zero dust, set up a two population model, else use a simple accretion disc
    if model['disc']['d2g'] > 0:
        try:
            disc = DustGrowthTwoPop(grid, star, eos, p['d2g'], model['dust']['radii_thresholds'], Sigma=Sigma, Sc=model['disc']['Schmidt'], feedback=feedback, uf_ice=model['dust']['ice_frag_v'], distribution_slope=model['dust']['p'])
        except:
            disc = DustGrowthTwoPop(grid, star, eos, p['d2g'], model['dust']['radii_thresholds'], Sigma=Sigma, Sc=model['disc']['Schmidt'], feedback=feedback, uf_ice=model['dust']['ice_frag_v'])
    else:
        disc = AccretionDisc(grid, star, eos, Sigma=Sigma)

    # Setup the UV irradiation
    try:
        p = model['fuv']
        disc.set_FUV(p['fuv_field'])
    except KeyError:
        p = model['uv']
        disc.set_FUV(p['uv_field'])

    return disc


def setup_model(model, disc, start_time=0, t_out = None):
    '''Setup the physics of the model'''
    
    gas       = None
    dust      = None
    diffuse   = None
    chemistry = None

    if model['transport']['gas']:
        try:
            gas = ViscousEvolution(boundary=model['grid']['outer_bound'], in_bound=model['grid']['inner_bound'])
        except KeyError:
            print("Default boundaries")
            gas = ViscousEvolution(boundary='Mdot_out')
        
    if model['transport']['diffusion']:
        diffuse = TracerDiffusion(Sc=model['disc']['Schmidt'])
    if model['transport']['radial drift']:
        dust = SingleFluidDrift(diffuse)
        diffuse = None

    # Inititate the correct external photoevaporation routine
    # FRIED should be considered default 
    try:
        p = model['fuv']
    except KeyError:
        p = model['uv']
    if (p['photoevaporation'] == "Constant"):
        photoevap = photoevaporation.FixedExternalEvaporation(disc, Mdot=1e-9)
    elif (p['photoevaporation'] == "FRIED" and disc.FUV>0):
        photoevap = photoevaporation.FRIEDExternalEvaporationMS(disc) # Using 2DMS at 400
    elif (p['photoevaporation'] == "FRIED" and disc.FUV<=0):
        photoevap = None
    elif (p['photoevaporation'] == "Integrated"):
        photoevap = photoevaporation.FRIEDExternalEvaporationM(disc) # Using integrated M(<R), extrapolated to M400
    elif (p['photoevaporation'] == "None"):
        photoevap = None
    else:
        print("Photoevaporation Mode Unrecognised: Default to 'None'")
        photoevap = None

    # Add internal photoevaporation
    try:
        if (model['x-ray']['L_X'] > 0):
            internal_photo = PrimordialDisc(disc)
        else:
            internal_photo = None
    except KeyError:
        internal_photo = None

    return DiscEvolutionDriver(disc, 
                               gas=gas, dust=dust, diffusion=diffuse, ext_photoevaporation=photoevap, int_photoevaporation=internal_photo,
                               t0=start_time, t_out=t_out)

def setup_output(model):
    
    out = model['output']

    # For explicit control of output times
    if (out['arrange'] == 'explicit'):

        # Setup of the output controller
        output_times = np.arange(out['first'], out['last'], out['interval']) * yr
        if not np.allclose(out['last'], output_times[-1], 1e-12):
            output_times = np.append(output_times, out['last'] * yr)

        # Setup of the plot controller
        if out['plot']:
            plot = np.array(out["plot_times"]) * yr
        else:
            plot = []

    # For regular, logarithmic output times
    elif (out['arrange'] == 'log'):

        # Setup of the output controller
        if out['interval']<10:
            perdec = 10
        else:
            perdec = out['interval']
        first_log = np.floor( np.log10(out['first']) * perdec ) / perdec
        last_log  = np.floor( np.log10(out['last'])  * perdec ) / perdec
        no_saves = (last_log-first_log)*perdec+1
        output_times = np.logspace(first_log,last_log,no_saves,endpoint=True,base=10,dtype=int) * yr
        output_times = np.insert(output_times,0,0)
        if not np.allclose(out['last'], output_times[-1], 1e-12):
            output_times = np.append(output_times, out['last'] * yr)

        # Setup of the plot controller
        if out['plot']:
            plot = output_times
        else:
            plot = []      

    EC = Event_Controller(save=output_times, plot=plot)
    
    # Base string for output:
    mkdir_p(out['directory'])
    #mkdir_p(out['directory']+'_profiles')
    base_name = os.path.join(out['directory'], out['base'] + '_{:04d}')

    format = out['format']
    if format.lower() == 'hdf5':
        base_name += '.h5'
    elif format.lower() == 'ascii':
        base_name += '.dat'
    else:
        raise ValueError ("Output format {} not recognized".format(format))

    return base_name, EC, output_times / yr


def setup_wrapper(model, restart, output=True):
    # Setup model
    disc = setup_disc(model)
    if restart:
        disc, time, datadict = restart_model(model, disc, restart)
        driver = setup_model(model, disc, time, t_out = datadict['t'])
    else:
        driver = setup_model(model, disc)

    if output:
        # Setup outputs
        output_name, io_control, output_times = setup_output(model)
        plot_name = model['output']['plot_name']
    else:
        output_name, io_control, plot_name = None, None, None

    # Truncate disc at base of wind
    if (driver.photoevap is not None):
        if (isinstance(driver.photoevap,photoevaporation.FRIEDExternalEvaporationMS)):
            driver.photoevap.optically_thin_weighting(disc)
            optically_thin = (disc.R > driver.photoevap._Rot)
        else:
            initial_trunk = photoevaporation.FRIEDExternalEvaporationMS(disc)
            initial_trunk.optically_thin_weighting(disc)
            optically_thin = (disc.R > initial_trunk._Rot)

        disc._Sigma[optically_thin] = 0
        disc._Rot = np.array([])

        """Lines to truncate with no mass loss if required for direct comparison"""
    """else:
        photoevap = photoevaporation.FRIEDExternalEvaporationMS(disc)
        optically_thin = (disc.R > disc.Rot(photoevap))"""
    
    Dt_nv = np.zeros_like(disc.R)
    if (driver.photoevap is not None):
        # Perform estimate of evolution for non-viscous case
        (_, _, M_cum, Dt_nv) = driver.photoevap.get_timescale(disc)

    return disc, driver, output_name, io_control, plot_name, Dt_nv

def restart_model(model, disc, snap_number):
    # Resteup model
    out = model['output']
    reader = DiscReader(out['directory'], out['base'], out['format'])

    snap = reader[snap_number]

    disc.Sigma[:] = snap.Sigma
    try:
        disc.dust_frac[:] = snap.dust_frac
        disc.grain_size[:] = snap.grain_size
    except:
        pass

    time = snap.time * yr       # Convert real time (years) to code time

    disc.update(0)

    # Revise history
    try:
        inputdata = np.loadtxt(model['output']['directory']+"/"+"discproperties.dat")
        infile = open(model['output']['directory']+"/"+"discproperties.dat", 'r')
    except:
        inputdata = np.loadtxt(model['output']['directory']+"/"+model['output']['directory']+"_discproperties.dat")
        infile = open(model['output']['directory']+"/"+"discproperties.dat", 'r')

    # Data headers
    for line in infile:
        head=line
        break
    infile.close()
    head = head.split("\t")
    head[0]  = head[0].split("# ")[-1]
    head[-1] = head[-1].split("\n")[0]

    datadict = {}
    for h in range(0,len(head)):
        datadict[head[h]] = inputdata[:snap_number+1,h]

    # Rewrite history
    not_future = disc.history.restart(datadict, time/yr)    # Pass time in years
    print("Restarting with times:")
    print(datadict['t'][not_future])

    return disc, time, datadict     # Return disc objects, input data and time in code units

###############################################################################
# Saving
###############################################################################

def save_summary(driver,model,):
    # 0 Select times of recording
    used_times = driver._output_times
    dust = isinstance(driver.disc,DustGrowthTwoPop)

    # 1 Retrieve radii
    outer_radii, scale_radii, ot_radii, hole_radii = driver.disc.history.radii()
    radii_select = {}
    if driver.photoevap:
        radii_select['R_out'] = ot_radii
    else:
        radii_select['R_out'] = outer_radii
    if np.isnan(scale_radii).sum() < len(scale_radii):
        radii_select['R_C'] = scale_radii
    if np.isnan(hole_radii).sum() < len(hole_radii):
        radii_select['R_hole'] = hole_radii

    # 2 Retrieve masses
    disc_masses = driver.disc.history.mass()

    # 3 Dust
    if (isinstance(driver.disc,DustGrowthTwoPop)):
        dust_masses, dust_wind = driver.disc.history.mass_dust()
        dust_radii = driver.disc.history.radii_dust()

    # 4 Accretion rates
    Macc, Mext, Mint = driver.disc.history.mdot()

    # 5 Photoevaporation rates
    Mevap = {}
    if driver.photoevap:
        Mevap['M_ext'] = Mext
    if driver._internal_photo:
        Mevap['M_int'] = Mint

    # Save data
    outputdata = np.column_stack((used_times, disc_masses))
    head  = ['t','M_D']
    units = ['yr','g']
    if dust:
        outputdata = np.column_stack((outputdata, dust_masses))
        head.append('M_d')
        units.append('g')

    for key, radii in radii_select.items():
        outputdata = np.column_stack((outputdata, radii))
        head.append(key)
        units.append('AU')
    if dust:
        for key, radii in dust_radii.items():
            outputdata = np.column_stack((outputdata, radii))
            head.append('R_{}'.format(int(float(key)*100)))
            units.append('AU')

    if driver._gas:
        outputdata = np.column_stack((outputdata, Macc))
        head.append('M_acc')
        units.append('Msun/yr')

    for key, mdot in Mevap.items():
        outputdata = np.column_stack((outputdata, mdot))
        head.append(key)
        units.append('Msun/yr')
    if dust and driver.photoevap:
        outputdata = np.column_stack((outputdata, dust_wind))
        head.append('M_wind')
        units.append('g')

    head  = "\t".join(head)
    units = "\t".join(units)
    full_head = "\n".join([head,units])
        
    np.savetxt(model['output']['directory']+"/"+"discproperties.dat", outputdata, delimiter='\t', header=full_head)

    return outputdata

###############################################################################
# Run
###############################################################################    

def run(model, io, base_name, plot_name, mass_loss_mode, all_in, restart, verbose=True, n_print=1000, end_low=False):
    end = False     # Flag to set in order to end computation
    hole_open = 0   # Flag to set to snapshot hole opening
    hole_save = 0   # Flag to set to snapshot hole opening

    if restart:
        # Skip evolution already completed
        while not io.finished():
            ti = io.next_event_time()
            
            if ti > model.t:
                break
            else:
                io.pop_events(model.t)

    while not io.finished():
        ti = io.next_event_time()
        while (model.t < ti and end==False):
            """The model can break down when all at base rate because i_max = i_edge, Sigma(i_edge -> 0). Instead, terminate the model"""
            """Moreover, the rates cannot be trusted to be physical by this point"""
            """Also want to stop when reach unobservably low accretion rates"""
            """Finally, internally photoevaporating models must stop if the disc is empty"""
            if model.photoevap:
                # Read mass loss rates
                not_empty = (model.disc.Sigma_G > 0)
                Mdot_evap = model.photoevap.mass_loss_rate(model.disc,not_empty)
                # If at base evaporation rates
                if (np.amax(Mdot_evap)<=1e-10):
                    # Stop
                    print ("Photoevaporation rates below FRIED floor... terminating calculation at ~ {:.0f} yr".format(model.t/yr))
                    end = True
            if model._internal_photo:
                if model._internal_photo._empty:
                    # Stop
                    print ("No valid Hole radius as disc is depleted... terminating calculation at ~ {:.0f} yr".format(model.t/yr))
                    end = True
                elif model._internal_photo._switch:
                    hole_open = np.inf
                elif model._internal_photo._reset:
                    hole_open = 0
                    model._internal_photo._reset = False
                if model._internal_photo._Hole:
                    hole_open += 1
                    if (hole_open % 100000) == 1:
                        ti = model.t
                        break
            if model._gas and end_low:
                # If below observable accretion rates
                M_visc_out = 2*np.pi * model.disc.grid.Rc[0] * model.disc.Sigma[0] * model._gas.viscous_velocity(model.disc)[0] * (AU**2)
                Mdot_acc = -M_visc_out*(yr/Msun)
                if (Mdot_acc<1e-11):
                    # Stop
                    print ("Accretion rates below observable limit... terminating calculation at ~ {:.0f} yr".format(model.t/yr))
                    end = True
                    
            if end:
                last_save=0
                last_plot=0
                # If there are save times left
                if (np.size(io.event_times('save'))>0):
                    last_save = io.event_times('save')[-1]
                # If there are plot times left 
                if (np.size(io.event_times('plot'))>0):
                    last_plot = io.event_times('plot')[-1]
                # Remove all events up to the end
                last_t = max(last_save,last_plot)
                io.pop_events(last_t)
            else:
            ### Evolve model and return timestep ###
                dt = model(ti)

            ### Printing
            if verbose and (model.num_steps % n_print) == 0:
                print('Nstep: {}'.format(model.num_steps))
                print('Time: {} yr'.format(model.t / yr))
                print('dt: {} yr'.format(dt / yr))
                try:
                    print("Column density to hole is N = {} g cm^-2".format(model._internal_photo._N_hole))
                    print("Empty cells: {}".format(np.sum(model.disc.Sigma_G<=0)))
                    #print("Total mdot: {}".format(model._internal_photo._Mdot_true))
                except AttributeError:
                    pass
        grid = model.disc.grid
        
        ### Saving
        if (io.check_event(model.t, 'save') or end or (hole_open % 100000)==1):
            model._output_times.append(model.t / yr)
            save_no = len(model._output_times) - 1

            # Print message to record this
            if (hole_open % 100000)==1:
                print ("Taking extra snapshot of properties while hole is clearing")
                hole_save+=1
            elif end:
                print ("Taking snapshot of final disc state")
            else:
                print ("Making save at {} yr".format(model.t/yr))
            if base_name.endswith('.h5'):
                    model.dump_hdf5( base_name.format(save_no))
            else:
                    model.dump_ASCII(base_name.format(save_no))

            ### Measure disc properties and record

            # 1 Radius
            if (model.photoevap or model._internal_photo):
                model.disc.Rout(Track = True)
            else:
                model.disc.Rout(fit_LBP=True, Track=True) # Locate outer radius by the density threshold

            # 2 Disc Mass
            model.disc.Mtot(Track=True) # Total disc mass

            # 3 Dust radii and mass and wind loss mass
            if (isinstance(model.disc,DustGrowthTwoPop)):
                model.disc.Rdust(Track=True) # Radius containing proportion of dust mass
                model.disc.Mdust(Track=True) # Remaining dust mass
                model.disc.Mwind(Track=True) # Total mass of dust lost in wind (NB can be 0 if no photoevaproation)

            # 4 Track total viscous accretion rate
            if model._gas:
                model.disc.Mdot(model._gas.viscous_velocity(model.disc)[0], Track=True)

            # 5 Photoevaporative mass loss
            if (mass_loss_mode == 'Integrated'):
                # Get the raw mass loss rates
                (_, _) = model.photoevap.unweighted_rates(model.disc, Track=True)
            elif (model.photoevap is not None):
                # Get the weighted mass loss rates
                (Mdot_evap, _) = model.photoevap.optically_thin_weighting(model.disc, Track=True)

            # 6 Internal_photoevaporation
            if model._internal_photo:
                model._internal_photo.get_Rhole(model.disc, Track=True)
                
            # Save state
            save_summary(model,all_in)

        ### Plotting
        if (io.check_event(model.t, 'plot')  or end==True):
            err_state = np.seterr(all='warn')

            print('Nstep: {}'.format(model.num_steps))
            print('Time: {} yr'.format(model.t / (2 * np.pi)))

            np.seterr(**err_state)

        io.pop_events(model.t)


def main():
    # Retrieve model from inputs
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default=DefaultModel, help='specify the model input json file')
    parser.add_argument("--restart", "-r", type=int, default=0, help='specify a save number from which to restart')
    parser.add_argument("--end", "-e", action="store_true", help='include in order to stop when below observable accretion rates')
    args = parser.parse_args()
    model = json.load(open(args.model, 'r'))
    
    # Do all setup
    disc, driver, output_name, io_control, plot_name, Dt_nv = setup_wrapper(model, args.restart)

    # Run model
    run(driver, io_control, output_name, plot_name, model['fuv']['photoevaporation'], model, args.restart, end_low=args.end)
        
    # Save disc properties
    outputdata = save_summary(driver,model)

if __name__ == "__main__":
    main()
    
