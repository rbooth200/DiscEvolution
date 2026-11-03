# rkl.py : Runge-Kutta-Legendre super-time-stepping (RKL2; Meyer, Balsara &
# Aslam 2014) for the parabolic parts of the disc evolution: the viscous gas
# step and the diffusion of the dust and of chemical tracers. With these
# super-stepped, the time-step is set by the dust drift, the growth and the
# advection of tracers with the gas.
#
#   rkl2_stages(dt, dt_expl)     number of stages for a step dt
#   rkl2(rate, Y0, dt, s)        RKL2 step of dY/dt = rate(Y)
#   rkl2_flux(flux, div, Y0, dt, s)
#                                the same for dY/dt = div(flux(Y)), also
#                                returning the time-integrated flux
#   ViscousEvolutionRKL          ViscousEvolutionFV with an RKL2 step; tracers
#                                are advected once with the integrated flux
#   RKLDiffusion                 TracerDiffusion with the step by RKL2
# DiscEvolutionDriver uses them as it uses the explicit classes: the choice of
# gas and diffusion objects sets which terms are super-stepped.
################################################################################
import numpy as np
from .viscous_evolution import ViscousEvolutionFV
from .diffusion import TracerDiffusion


def rkl2_stages(dt, dt_expl):
    """Smallest number of RKL2 stages s >= 2 with dt <= dt_expl (s^2 + s - 2)/4,
    for dt_expl the forward-Euler stable step"""
    s = int(np.ceil(0.5*(np.sqrt(9 + 16*dt/dt_expl) - 1) - 1e-12))
    return max(s, 2)


def _coefficients(s):
    """RKL2 coefficients mu_j, nu_j, mu~_j, gamma~_j (j = 1..s; index 0 unused)"""
    b = np.array([1/3., 1/3.] + [(j*j + j - 2)/(2.*j*(j + 1)) for j in range(2, s + 1)])
    a = 1 - b
    w1 = 4./(s*s + s - 2)
    mu, nu, mut, gat = (np.zeros(s + 1) for _ in range(4))
    mut[1] = b[1]*w1
    for j in range(2, s + 1):
        mu[j] = (2*j - 1)/j * b[j]/b[j - 1]
        nu[j] = -(j - 1)/j * b[j]/b[j - 2]
        mut[j] = mu[j]*w1
        gat[j] = -a[j - 1]*mut[j]
    return mu, nu, mut, gat


def rkl2(rate, Y0, dt, s):
    """One RKL2 step of dY/dt = rate(Y) over dt with s stages"""
    mu, nu, mut, gat = _coefficients(s)
    M0 = rate(Y0)
    Ym2, Ym1 = Y0, Y0 + mut[1]*dt*M0
    for j in range(2, s + 1):
        Y = mu[j]*Ym1 + nu[j]*Ym2 + (1 - mu[j] - nu[j])*Y0 + \
            dt*(mut[j]*rate(Ym1) + gat[j]*M0)
        Ym2, Ym1 = Ym1, Y
    return Ym1


def rkl2_flux(flux, div, Y0, dt, s):
    """One RKL2 step of dY/dt = div(flux(Y)), div linear, over dt with s stages.

    The recursion is linear in the stage rates, so it is applied to the fluxes
    instead: G_s is the flux integrated over the step, and the new state is
    Y0 + div(G_s) exactly (as the stage states are). Anything carried by the
    flux can then be moved once with G_s, consistently with Y.

    returns:
        Y, G_s
    """
    mu, nu, mut, gat = _coefficients(s)
    F0 = flux(Y0)
    Gm2, Gm1 = 0*F0, mut[1]*dt*F0
    for j in range(2, s + 1):
        G = mu[j]*Gm1 + nu[j]*Gm2 + dt*(mut[j]*flux(Y0 + div(Gm1)) + gat[j]*F0)
        Gm2, Gm1 = Gm1, G
    return Y0 + div(Gm1), Gm1


class ViscousEvolutionRKL(ViscousEvolutionFV):
    """ViscousEvolutionFV with an RKL2 super-step.

    The viscous flux (of Sigma_G nu X) is evaluated at each stage with the
    dust fraction and nu held fixed. Tracers are then advected once with the
    flux integrated over the step (rkl2_flux): conservative, consistent with
    the gas (a uniform tracer stays uniform) and positive while each cell
    loses less than its mass, which sets max_timestep.

    args:
        cfl : Courant number of the tracer advection, default 0.5
        others as ViscousEvolutionFV (tol: the explicit step used for the
        number of stages)
    """
    def __init__(self, cfl=0.5, **kwargs):
        super(ViscousEvolutionRKL, self).__init__(**kwargs)
        self._cfl = cfl
        self.stages = 0

    def explicit_timestep(self, disc):
        """The forward-Euler step of ViscousEvolutionFV"""
        return super(ViscousEvolutionRKL, self).max_timestep(disc)

    def max_timestep(self, disc):
        """Advective limit of the tracers with the current mass flux"""
        self._setup_grid(disc.grid)
        self._init_fluxes(disc)
        dS, dA = self._dS, self._dA
        # what upwinding (_tracer_fluxes) takes out of each cell, per unit Sigma
        out = (dA[1:]*np.maximum(-dS[1:], 0) + dA[:-1]*np.maximum(dS[:-1], 0))/self._dV
        return self._cfl*(disc.Sigma/(out + 1e-300)).min()

    def __call__(self, dt, disc, tracers=[], adv=[]):
        """One step of the viscous evolution (as ViscousEvolutionFV.__call__)"""
        self._setup_grid(disc.grid)
        Sigma0 = disc.Sigma.copy()
        f_gas = disc.Sigma_G/(Sigma0 + 1e-300)       # held fixed

        def flux(Sigma):
            self._init_fluxes(disc, Sigma*f_gas)
            return self._dS

        def div(G):
            return np.diff(self._dA*G)/self._dV

        self.stages = rkl2_stages(dt, self.explicit_timestep(disc))
        Sigma_new, G = rkl2_flux(flux, div, Sigma0, dt, self.stages)

        # Tracers: one upwind step with the mean flux G/dt (_tracer_fluxes;
        # (t Sigma0 + dt F_t)/Sigma_new, as ViscousEvolutionFV's update)
        self._dS = G/dt
        for t in (tracers + adv):
            if t is None:
                continue
            t[:] = (dt*self._tracer_fluxes(t) + t*Sigma0) / (Sigma_new + 1e-300)

        disc.Sigma[:] = Sigma_new


class RKLDiffusion(TracerDiffusion):
    """TracerDiffusion with the step super-stepped by RKL2: no time-step limit
    (max_timestep is infinite); each step takes as many stages as its own
    explicit limit needs. Schmidt numbers and the disc are held fixed over
    the step.

    args: as TracerDiffusion
    """
    def __init__(self, *args, **kwargs):
        super(RKLDiffusion, self).__init__(*args, **kwargs)
        self.stages = 0

    def max_timestep(self, disc, Sc=None):
        return np.inf

    def step(self, dt, disc, eps_i, Sc=None):
        """Diffuse eps_i [..., Ncells] in place over dt (one RKL2 step)"""
        dt_e = super(RKLDiffusion, self).max_timestep(disc, self._get_Schmidt(disc, Sc))
        self.stages = rkl2_stages(dt, dt_e)
        eps_i[...] = rkl2(lambda Y: self(disc, Y, Sc), eps_i, dt, self.stages)
