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
from DiscEvolution.constants import Msun, AU, yr
from DiscEvolution.grid import Grid
from DiscEvolution.star import SimpleStar
from DiscEvolution.eos  import IrradiatedEOS
from DiscEvolution.dust import DustGrowthTwoPop
from DiscEvolution.opacity import Tazzari2016, Zhu2012
from DiscEvolution.viscous_evolution import ViscousEvolutionFV
from DiscEvolution.dust import SingleFluidDrift
from DiscEvolution.diffusion import TracerDiffusion
from DiscEvolution.driver import DiscEvolutionDriver
from DiscEvolution.io import Event_Controller, DiscReader
from DiscEvolution.disc_utils import mkdir_p

from DiscEvolution.chemistry import (
    ChemicalAbund, MolecularIceAbund, SimpleCNOAtomAbund, SimpleCNOMolAbund,
    SimpleCNOChemOberg, TimeDepCNOChemOberg,
    EquilibriumCNOChemOberg,
    SimpleCNOChemMadhu, EquilibriumCNOChemMadhu
)

try:
    from DiscEvolution.chemistry.krome_chem import (
        KromeIceAbund, KromeGasAbund, KromeMolecularIceAbund, KromeChem, 
        UserDust2GasCallBack)
except ImportError:
    # UserDust2GasCallBack must have a definition for this file to compile,
    # but this will never be used if krome_chem is not used
    UserDust2GasCallBack = object

from DiscEvolution.photoevaporation import (
    FixedExternalEvaportation, TimeExternalEvaportation)

###############################################################################
# Global Constants
###############################################################################
DefaultModel = "DiscConfig.json"

###############################################################################
# Global Functions
###############################################################################
class KromeCallBack(UserDust2GasCallBack):
    """Call back function for KROME user routines.

    This class does two things:
        1) Sets a globally constant cosmic ray ionization rate
        2) Sets the dust-to-gas ratio at each time step
        
    args:
        cosmic_ray_rate : float, default=1e-17
            Ionization rate of H_2 due to cosmic rays (per second)
    """
    def __init__(self, cosmic_ray_rate=1e-17, grain_size=1e-5):
        super(KromeCallBack, self).__init__(grain_size)
        self._crate = cosmic_ray_rate
    def init_krome(self, krome):
        super(KromeCallBack, self).init_krome(krome)
        try:
            krome.lib.krome_set_user_crate(self._crate)
        except AttributeError:
            pass

class VariableAmax_KromeCallBack(KromeCallBack):
    def __init__(self, cosmic_ray_rate=1e-17, grain_size=1e-5):
        super(VariableAmax_KromeCallBack, self).__init__(cosmic_ray_rate, 
                                                         grain_size)

    def __call__(self, krome, T, rho, dust_frac, **kwargs):
        # Set grain size
        if 'grain_size' in kwargs:
            asize = (kwargs['grain_size']*self._asize)**0.5
            krome.lib.krome_set_user_asize(asize)

        # Set other parameters
        super(VariableAmax_KromeCallBack, self).__call__(krome, T, rho, 
                                                         dust_frac, **kwargs)
    
def setup_init_abund_krome(model):
    Ncell = model['grid']['N']

    gas = KromeGasAbund(Ncell)
    ice = KromeIceAbund(Ncell)
        
    abund = KromeMolecularIceAbund(gas,ice)

    abund.gas.data[:] = 0
    abund.ice.data[:] = 0

    init_abund = np.genfromtxt(model['chemistry']['abundances'],
                               names=True, dtype=('|S5', 'f8', 'f8'),
                               skip_header=1)

    for name, value in zip(init_abund['Species'], init_abund['Abundance']):
        if name in ice.species:
            value += abund.ice.number_abund(name)
            abund.ice.set_number_abund(name,value)
        elif name in gas.species:
            value += abund.gas.number_abund(name)
            abund.gas.set_number_abund(name, value)
        else:
            pass

    if model['chemistry']['normalize']:
        norm = 1 / (abund.gas.total_abund + abund.ice.total_abund)
        abund.gas.data[:] *= norm
        abund.ice.data[:] *= norm
        
    # Add dust
    abund.gas.data[:] *= (1-disc.dust_frac.sum(0))
    abund.ice.data[:] *= (1-disc.dust_frac.sum(0))
    abund.ice["grain"] = disc.dust_frac.sum(0)

    return abund


def get_simple_chemistry_model(model):
    chem_type = model['chemistry']['type']

    grain_size = 1e-5
    try:
        grain_size = model['chemistry']['fixed_grain_size']
    except KeyError:
        pass
    
    if chem_type == 'TimeDep':
        chemistry = TimeDepCNOChemOberg(a=grain_size)
    elif chem_type == 'Madhu':
        chemistry = EquilibriumCNOChemMadhu(fix_ratios=False, a=grain_size)
    elif chem_type == 'Oberg':
        chemistry = EquilibriumCNOChemOberg(fix_ratios=False, a=grain_size)
    elif chem_type == 'NoReact':
        chemistry = EquilibriumCNOChemOberg(fix_ratios=True, a=grain_size)
    else:
        raise ValueError("Unkown chemical model type")

    return chemistry
   
def setup_init_abund_simple(model, disc):
    chemistry = get_simple_chemistry_model(model)

    X_solar = SimpleCNOAtomAbund(model['grid']['N'])
    X_solar.set_solar_abundances()

    # Iterate as the ice fraction changes the dust-to-gas ratio
    for i in range(10):
        chem = chemistry.equilibrium_chem(disc.T,
                                          disc.midplane_gas_density,
                                          disc.dust_frac.sum(0),
                                          X_solar)
        disc.initialize_dust_density(chem.ice.total_abund)
    return chem

def setup_disc(model):
    '''Create disc object from initial conditions'''
    # Setup the grid, star and equation of state
    p = model['grid']
    grid = Grid(p['R0'], p['R1'], p['N'], spacing=p['spacing'])

    p = model['star']
    star = SimpleStar(M=p['mass'], R=p['radius'], T_eff=p['T_eff'])
    
    p = model['eos']
    if p['type'] == 'irradiated':
        if p['opacity'] == 'Tazzari2016':
            kappa = Tazzari2016()
        elif p['opacity'] == 'Zhu2012':
            kappa = Zhu2012
        else:
            raise ValueError("Opacity not recognised")
        
        eos = IrradiatedEOS(star, model['disc']['alpha'], kappa=kappa)
    elif p['type'] == 'iso':
        eos = LocallyIsothermalEOS(star, p['h0'], p['q'], 
                                   model['disc']['alpha'])
    else:
        raise ValueError("Error: eos::type not recognised")
    eos.set_grid(grid)
    
    # Setup the physical part of the disc
    p = model['disc']
    Sigma = np.exp(-grid.Rc / p['Rc']) / (grid.Rc)
    Sigma *= p['mass'] / np.trapz(Sigma, np.pi*grid.Rc**2)
    Sigma *= Msun / AU**2

    eos.update(0, Sigma)

    try:
        feedback = model['disc']['feedback']
    except KeyError:
        feedback = True

    if model['disc']['d2g'] > 0:
        amin = model['disc']['amin']

        disc = DustGrowthTwoPop(grid, star, eos, p['d2g'], Sigma=Sigma, 
                                amin=amin, Sc=model['disc']['Schmidt'], 
                                f_grow=model['disc'].get('f_grow',1.0),
                                feedback=feedback)
    else:
        disc = AccretionDisc(grid, star, eos, Sigma)

    # Setup the chemical part of the disc
    if model['chemistry']["on"]:
        if model['chemistry']['type'] == 'krome':
            disc.chem = setup_init_abund_krome(model)
            disc.update_ices(disc.chem.ice)
        else:
            disc.chem =  setup_init_abund_simple(model, disc)
            disc.update_ices(disc.chem.ice)

    return disc


def setup_krome_chem(model):
    if model['chemistry']['fix_mu']:
        mu = model['chemistry']['mu']
    else:
        mu = 0.

    crate = 1e-17
    try:
        crate = model['chemistry']['crate']
    except KeyError:
        pass
    grain_size = 1e-5
    try:
        grain_size = model['chemistry']['fixed_grain_size']
    except KeyError:
        pass
    
    call_back = KromeCallBack(crate, grain_size)
    try:
        if model['chemistry']['variable_grain_size']:
            call_back = VariableAmax_KromeCallBack(crate, grain_size)
    except KeyError:
        pass

    chemistry = KromeChem(renormalize=model['chemistry']['normalize'],
                          fixed_mu=mu, call_back=call_back)

    return chemistry

def setup_simple_chem(model):
    return get_simple_chemistry_model(model)

def setup_model(model, disc, start_time):
    '''Setup the physics of the model'''
    
    gas       = None
    dust      = None
    diffuse   = None
    chemistry = None
    photoevap = None

    if model['transport']['gas']:
        gas = ViscousEvolutionFV()
    if model['transport']['diffusion']:
        diffuse = TracerDiffusion(Sc=model['disc']['Schmidt'])
    if model['transport']['radial drift']:
        van_leer = model['dust_transport']['van leer']
        settling = model['dust_transport']['settling']
        
        if model['dust_transport']['diffusion']:
            dust_diffusion = diffuse
            diffuse = None
        else:
            dust_diffusion = None

        dust = SingleFluidDrift(diffusion=dust_diffusion, 
                                settling=settling,
                                van_leer=van_leer)
    if model['photoevaporation']['on']:
        if model['photoevaporation']['method'] == 'const':
            photoevap = \
                FixedExternalEvaportation(model['photoevaporation']['coeff'])
        else:
            photoevap = \
                TimeExternalEvaportation(model['photoevaporation']['coeff'])

    if model['chemistry']['on']:
        if  model['chemistry']['type'] == 'krome':
            chemistry = setup_krome_chem(model)
        else:
            chemistry = setup_simple_chem(model)


    return DiscEvolutionDriver(disc, 
                               gas=gas, dust=dust, diffusion=diffuse,
                               chemistry=chemistry,
                               photoevaporation=photoevap,
                               t0=start_time)

def restart_model(model, disc, snap_number):
    
    out = model['output']
    reader = DiscReader(out['directory'], out['base'], out['format'])

    snap = reader[snap_number]

    disc.Sigma[:] = snap.Sigma
    disc.dust_frac[:] = snap.dust_frac
    disc.grain_size[:] = snap.grain_size

    time = snap.time * yr

    try:
        chem = snap.chem
        disc.chem.gas.data[:] = chem.gas.data
        disc.chem.ice.data[:] = chem.ice.data
    except AttributeError as e:
        if model['chemistry']['on']:
            raise e

    disc.update(0)

    return disc, time



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

def _plot_grid(model):
    grid = model.disc.grid 

    try:
        eps = model.disc.dust_frac.sum(0)
        plt.subplot(222)
        plt.loglog(grid.Rc, eps)
        plt.xlabel('$R$')
        plt.ylabel('$\epsilon$')
        plt.ylim(ymin=1e-4)
        plt.subplot(223)
        plt.loglog(grid.Rc, model.disc.Stokes()[1])
        plt.xlabel('$R$')
        plt.ylabel('$St$')
        plt.subplot(224)
        plt.loglog(grid.Rc, model.disc.grain_size[1])
        plt.xlabel('$R$') 
        plt.ylabel('$a\,[\mathrm{cm}]$')

        plt.subplot(221)
        l, = plt.loglog(grid.Rc, model.disc.Sigma_D.sum(0), '--')
        c = l.get_color()
    except AttributeError:
        c = None

    plt.loglog(grid.Rc, model.disc.Sigma_G, c=c)
    plt.xlabel('$R$')
    plt.ylabel('$\Sigma_\mathrm{G, D}$')
    plt.ylim(ymin=1e-5)


def run(model, io, base_name, restart, verbose=True, n_print=100):

    if restart:
        # Skip evolution already completed
        while not io.finished():
            ti = io.next_event_time()
            
            if ti > model.t:
                break
            else:
                io.pop_events(model.t)

        assert io.event_number('save') == restart+1

    plot = False
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
            plot = True
            err_state = np.seterr(all='warn')

            print('Nstep: {}'.format(model.num_steps))
            print('Time: {} yr'.format(model.t / (2 * np.pi)))
            
            _plot_grid(model)

            np.seterr(**err_state)

        io.pop_events(model.t)

    if plot:
        plt.show()

def main():
    
    err_state = np.seterr(invalid='raise')

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default=DefaultModel)
    parser.add_argument("--restart", "-r", type=int, default=0)

    args = parser.parse_args()

    model = json.load(open(args.model, 'r'))
    
    disc = setup_disc(model)

    if args.restart: 
        disc, time = restart_model(model, disc, args.restart)
    else:
        time = 0 

    driver = setup_model(model, disc, time)

    output_name, io_control = setup_output(model)

    run(driver, io_control, output_name, args.restart)

    np.seterr(**err_state)

if __name__ == "__main__":
    main() 



