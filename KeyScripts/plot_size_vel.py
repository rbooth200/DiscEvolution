from __future__ import print_function
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import rcParams
rcParams['image.cmap'] = 'plasma'
from scripts.snap_reader import DiscReader
from DiscEvolution.constants import *
from DiscEvolution.disc_utils import mkdir_p
from DiscEvolution.viscous_evolution import ViscousEvolution
import KeyScripts.run_model as run_model
import sys
import argparse
import json

class Formatter(object):
    def __init__(self, im):
        self.im = im
    def __call__(self, x, y):
        z = self.im.get_array()[int(y), int(x)]
        return 'x={:.01f}, y={:.01f}, z={:.01f}'.format(x, y, z)

def load_model(model_no):
    loadfile = "DiscConfig"+str(model_no)+".json"
    model = json.load(open(loadfile, 'r'))

    DIR = model['output']['directory']
    alpha = model['disc']['alpha']
    plot_times = np.array(model['output']['plot_times'])

    return DIR, alpha, plot_times, model

def Facchini_limit(disc, Mdot, rho_s=1, M_star=1):
    """
    Equation 35 of Facchini et al (2016)
    Note following definitions:
    F = H / sqrt(H^2+R^2) (dimensionless)
    v_th = \sqrt(8/pi) C_S in AU / t_dyn
    Mdot is in units of Msun yr^-1
    G=1 in units AU^3 Msun^-1 t_dyn^-2
    """
    
    F = disc.H / np.sqrt(disc.H**2+disc.R**2)
    rho = rho_s
    Mstar = M_star
    v_th = np.sqrt(8/np.pi) * disc.cs
        
    a_entr = (v_th * Mdot) / (Mstar * 4 * np.pi * F * rho)
    a_entr *= Msun / AU**2 / yr
    return a_entr

def Stokes(disc, rho_s=1):
    return (np.pi * rho_s) / 2 * disc.grain_size[1,:] / (disc.Sigma*(1-disc.dust_frac.sum(0)) + 1e-300)

if __name__ ==  "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", "-m", nargs='+', type=int, default="_default")
    args = parser.parse_args()
    models = args.models

    # Setup array of models and read disc state
    reader = np.array([])
    alpha = np.array([])
    M_dot_w = []
    R_disc = []
    N_snaps = []
    for m in models:
        DIR, alpha_m, plot_times, model = load_model(m)       
        reader = np.append(reader,DiscReader(DIR, 'disc'))
        alpha = np.append(alpha,alpha_m)

        # Read saved summary properties
        inputdata = np.loadtxt(DIR+"/"+DIR+"_discproperties.dat")
        R_disc.append(inputdata[:,2].flatten())
        M_dot_w.append(inputdata[:,4].flatten())

        # Count Snaps
        N_snaps.append(reader[-1].Num_Snaps)

    snaps = range(0, min(N_snaps))
    
    plot_type = "size"
    #plot_type = "flux"

    out_dir = plot_type+"_comparison_{}".format(models)
    mkdir_p(out_dir)

    if (plot_type=="size"):
        for n in snaps:
            plt.figure()
            plt.rcParams['text.usetex'] = "True"
            plt.rcParams['font.family'] = "serif"

            for i in range(0,len(models)):
                disc = reader[i][n]
                disc.cs =  (GasConst * disc.T / 2.4)**0.5 / (AU * Omega0)
                disc.H = ((disc.cs) * disc.R**1.5)

                plt.subplot(2,1,1)
                p = plt.loglog(disc.R, disc.grain_size[1,:], label="Model {} ($a_{{max}}$)".format(models[i]))
                #plt.title("Model {}".format(models[i]))
                a_ent = Facchini_limit(disc,M_dot_w[i][n])
                plt.loglog(disc.R, a_ent, linestyle='--', color=p[0].get_color(), label="Model {} ($a_{{ent}}$)".format(models[i]))
                #plt.xlabel('$R~/~\mathrm{AU}$',fontsize=13)
                plt.ylabel('$a~/~\mathrm{cm}$',fontsize=13)

                plt.subplot(2,1,2)
                plt.loglog(disc.R,Stokes(disc))
                plt.ylim([1e-10,1e1])
                plt.xlabel('$R~/~\mathrm{AU}$',fontsize=13)
                plt.ylabel('$St$',fontsize=13)
            
            plt.subplot(2,1,1)
            plt.legend()
            cur_t = plot_times[n]
            if (cur_t>0):    
                logt = np.log10(cur_t)
                exponent = int(logt)
                prefactor = np.power(10,logt-exponent)
                plt.title('Time: $t={:.2f}\\times10^{:d}~\mathrm{{yr}}$'.format(prefactor,exponent), fontsize=16)
            else:
                plt.title('Time: $t=0~\mathrm{{yr}}$', fontsize=16)
            
            plt.savefig(out_dir+"/size_comparison_{}.png".format(n))
            plt.close()

    elif (plot_type=="flux"):
        gas = ViscousEvolution(boundary='Mdot')
        disc2 = run_model.setup_disc(model)

        for n in snaps:
            plt.figure()
            plt.rcParams['text.usetex'] = "True"
            plt.rcParams['font.family'] = "serif"

            for i in range(0,len(models)):
                disc = reader[i][n]
                disc.grid = disc2.grid
                disc.cs =  (GasConst * disc.T / 2.4)**0.5 / (AU * Omega0)
                disc.H = ((disc.cs) * disc.R**1.5)
                h = disc.H / disc.R
                disc.nu = alpha[i] * disc.cs * disc.H
                disc.Sigma_G = disc.Sigma * (1-disc.dust_frac.sum(0))
                mdot = (3*np.pi * disc.nu * disc.Sigma) * AU**2 * Omega0
                mdot /= Msun / (365.25 * 3600*24)

                St = Stokes(disc)
                Omega_k = Omega0 * disc.R**(-1.5)
                v_k = disc.R * Omega_k
                rho_mid = disc.Sigma / (np.sqrt(2*np.pi) * disc.H * AU)
                P = rho_mid * disc.cs**2
                R = disc.R
                gamma = np.empty_like(P)
                gamma[1:-1] = abs((P[2:] - P[:-2])/(R[2:] - R[:-2]))
                gamma[ 0]   = abs((P[ 1] - P[  0])/(R[ 1] - R[ 0]))
                gamma[-1]   = abs((P[-1] - P[ -2])/(R[-1] - R[-2]))
                gamma *= R/(P+1e-300)
                u_drift = disc.cs**2 / v_k * gamma / (St+St**(-1)) * Omega0
                u_visc = abs(gas.viscous_velocity(disc))
                u_tl = 3 * alpha[i] * disc.cs**2 / v_k /2 * Omega0

                p = plt.loglog(disc.grid.Re[1:-1], 2*np.pi*u_visc, label="Model {}, Visc".format(models[i]))
                plt.loglog(disc.R, 2*np.pi*u_drift, linestyle='--', color=p[0].get_color(), label="Model {}, Drift".format(models[i]))

                plt.loglog([R_disc[i][n],R_disc[i][n]],[1e-8,1], linestyle=':', color=p[0].get_color())
                plt.ylim([1e-4,1e-1])
                plt.xlim([40,150])
                plt.xlabel('$R~/~\mathrm{AU}$',fontsize=13)
                plt.ylabel('Velocity / $\mathrm{AU~yr}^{-1}$')
            
            plt.legend()
            cur_t = plot_times[n]
            if (cur_t>0):    
                logt = np.log10(cur_t)
                exponent = int(logt)
                prefactor = np.power(10,logt-exponent)
                plt.title('Time: $t={:.2f}\\times10^{:d}~\mathrm{{yr}}$'.format(prefactor,exponent), fontsize=16)
            else:
                plt.title('Time: $t=0~\mathrm{{yr}}$', fontsize=16)
            
            plt.savefig(out_dir+"/flux_comparison_{}_zoom.png".format(n))
            plt.close()

    else:
        print("Plot type not recognised.")
