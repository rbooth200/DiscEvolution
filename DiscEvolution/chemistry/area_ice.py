# area_ice.py
#
# Thermal adsorption/desorption for the TriMom dust model
# (dust_trimom.DustGrowthTriMom) with the ice of each volatile in two parts:
# a mass-distributed part, inside the grains, and a shell, a mantle of
# uniform thickness d on every grain.
#
# Per unit grain surface, adsorption (~ the vapour) and multilayer desorption
# (zeroth order) change every grain's radius at the same rate, so a mantle of
# uniform thickness stays uniform: the ice that condenses is a shell, one
# number per volatile and cell. Its thickness follows from its mass on
# TriMom's power law (dust_trimom.shell_thickness), and it adsorbs and
# desorbs on the coated area. The equilibria are those of ThermalChem.
#
# Each step is backward Euler for every grain, in closed form on the power
# law. The grains are the dust less the shells: size a, with the volatile's
# mass-distributed ice uniform by mass (core c = kappa a). A grain's ice
# thickness t = a + d - c after the step is linear in its start for a given
# end vapour: t + gain - loss above a monolayer, (t + gain)/(1 + loss) below
# one (first-order desorption). So the ice is a closed form of the vapour,
# and the vapour is a monotone root. New ice goes into the shell. Losses come
# out of the shell, then out of the mass-distributed part (peeling, which
# strips the small grains first). The sub-monolayer equilibrium coverage of
# the cores goes into the shell (it moves with the area).
#
# Transport (MomentDrift(area_tracers=...)) moves each shell at its
# mass-weighted velocity with the Stokes number of the coated size, and the
# mass-distributed ice with the grains. Growth (DustGrowthTriMom) mixes the
# shells into the grains (mix_area_to_mass), which grow with it if
# ice_sizes=True. Write-up: shell_ice.tex in scripts/trimom/production/ice_area/.
################################################################################
import numpy as np

from ..constants import m_H
from ..dust_trimom import moment_ratio, shell_thickness
from .base_chem import ChemicalAbund, ThermalChem, SimpleChemBase


class AreaMassIceAbund(ChemicalAbund):
    """Ice abundances with the ice of the volatiles in area_species split
    into a shell (rows '<spec>:A'), a mantle of uniform thickness on the
    grains, and a mass-distributed part (rows '<spec>'). The other species
    (e.g. refractory grains) are distributed by mass.

    ice[spec] is the total; setting it keeps the shell (up to the new
    total). data holds all the rows, which the driver transports; area_rows
    are the indices to pass to MomentDrift(area_tracers=...).

    args:
        species      : list of species names
        masses       : their molecular masses in amu
        area_species : the species with a shell
        sizes        : shape of the data for each species
    """
    def __init__(self, species, masses, area_species, *sizes):
        species, masses = list(species), list(masses)
        rows = species + [s + ':A' for s in area_species]
        row_masses = masses + [masses[species.index(s)] for s in area_species]
        super(AreaMassIceAbund, self).__init__(rows, row_masses, *sizes)
        self._species = species
        self._area_species = list(area_species)

    @classmethod
    def from_ice(cls, ice, area_species):
        """All the ice of an existing ChemicalAbund, distributed by mass"""
        new = cls(ice.names, ice.masses, area_species, ice.size)
        new._data[:len(ice.names)] = ice.data
        return new

    def __getitem__(self, k):
        if k in self._area_species:
            return self.mass_part(k) + self.area(k)
        return self._data[self._indexes[k]]

    def __setitem__(self, k, val):
        if k in self._area_species:
            A = np.minimum(self.area(k), val)
            self._data[self._indexes[k + ':A']] = A
            self._data[self._indexes[k]] = val - A
        else:
            self._data[self._indexes[k]] = val

    def __iter__(self):
        return iter(self._species)

    @property
    def names(self):
        return self._species

    @property
    def species(self):
        return self._species

    @property
    def area_species(self):
        return self._area_species

    @property
    def area_rows(self):
        """Indices of the shell rows in data"""
        return np.array([self._indexes[s + ':A'] for s in self._area_species])

    @property
    def mass_rows(self):
        """Indices of the mass-distributed rows of the area_species in data"""
        return np.array([self._indexes[s] for s in self._area_species])

    def area(self, k):
        """The ice in the shell of species k"""
        return self._data[self._indexes[k + ':A']]

    def mass_part(self, k):
        """Mass-distributed ice of species k"""
        return self._data[self._indexes[k]]

    def set_parts(self, k, area, mass):
        """Set the shell and mass-distributed ice of species k"""
        self._data[self._indexes[k + ':A']] = area
        self._data[self._indexes[k]] = mass

    def mix_area_to_mass(self, f):
        """Move a fraction f of the shells into the mass-distributed ice
        (growth mixing)"""
        iA, iM = self.area_rows, self.mass_rows
        dA = f*self._data[iA]
        self._data[iM] += dA
        self._data[iA] -= dA

    def clip_negative(self):
        """Set negative abundances to zero. For the area_species, a negative
        part is taken out of the other one, keeping the total; a negative
        total empties both."""
        iA, iM = self.area_rows, self.mass_rows
        rest = np.setdiff1d(np.arange(len(self._data)), np.concatenate([iA, iM]))
        self._data[rest] = np.maximum(self._data[rest], 0)
        A, M = self._data[iA], self._data[iM]
        X = np.maximum(A + M, 0)
        A = np.clip(A, 0, X)
        self._data[iA] = A
        self._data[iM] = X - A


def _frac_below(k, x, L):
    """int_0^x e^(k y) dy / int_0^L e^(k y) dy, for 0 <= x <= L, without
    overflow"""
    small = np.abs(k*L) < 1e-8
    ks = np.where(small, 1., k)
    with np.errstate(over='ignore', invalid='ignore'):
        up = np.exp(ks*(x - L))*np.expm1(-ks*x)/np.expm1(-ks*L)
        down = np.expm1(ks*x)/np.expm1(ks*L)
    return np.where(small, x/L, np.where(ks > 0, up, down))


def _newton_root(fdf, x0, lo, hi, tol, max_iter=100):
    """Roots of increasing functions f in [lo, hi] (f(lo) <= 0 <= f(hi)),
    vectorised: Newton's method from x0, bisecting when a step leaves the
    bracket, on the unconverged cells only (fdf(x, idx) gives f and f' at x
    for the cells idx). Converged when |f| <= tol, or the step or the bracket
    is at round-off.

    returns: the roots and the converged mask"""
    x, lo, hi = np.clip(x0, lo, hi), lo.copy(), hi.copy()
    done = np.zeros(len(x), dtype=bool)
    idx = np.arange(len(x))
    for _ in range(max_iter):
        xi = x[idx]
        f, df = fdf(xi, idx)
        pos = f > 0
        hi[idx] = np.where(pos, xi, hi[idx])
        lo[idx] = np.where(pos, lo[idx], xi)
        with np.errstate(divide='ignore', invalid='ignore'):
            xn = xi - f/df
        inside = np.isfinite(xn) & (xn > lo[idx]) & (xn < hi[idx])
        xn = np.where(inside, xn, 0.5*(lo[idx] + hi[idx]))
        conv = (np.abs(f) <= tol[idx]) | (np.abs(xn - xi) <= 1e-15*np.abs(xi)) | \
            (hi[idx] - lo[idx] <= 1e-15*np.abs(hi[idx]))
        done[idx[conv]] = True
        x[idx] = np.where(conv, xi, xn)
        idx = idx[~conv]
        if not len(idx):
            break
    return x, done


class TimeDependentChemTriMom(ThermalChem, SimpleChemBase):
    """Time dependent thermal adsorption/desorption with the ice in shells
    and mass-distributed parts (AreaMassIceAbund), for the TriMom dust model.

    Uses the size distribution of disc (a DustGrowthTriMom) and the rates of
    ThermalChem, on the area of the grains with their shells. The grains are
    the dust less all the shells (dust_frac less the shell rows). The
    chemistry is a separate operator: transport moves the two parts with
    their own velocities and growth mixes them (MomentDrift(area_tracers=...),
    DustGrowthTriMom).

    Each step is backward Euler for every grain of the power law, in closed
    form, with the vapour at the end of the step a monotone root. It is exact
    in equilibrium, for multilayer desorption into empty gas, and in the
    sub-monolayer regime's linear decay. New ice goes into the shell; losses
    come out of the shell, then out of the mass-distributed part (peeling,
    the small grains bare). The equilibrium sub-monolayer coverage of the
    cores is kept in the shell.

    n_unconverged is the number of cells whose vapour did not converge in
    the last step.

    args:
        disc : the DustGrowthTriMom disc
        tol  : relative tolerance of the vapour, default 1e-12
        other args as ThermalChem
    """
    def __init__(self, disc, tol=1e-12, **kwargs):
        ThermalChem.__init__(self, **kwargs)
        SimpleChemBase.__init__(self)
        self._disc = disc
        self._tol = tol
        self.n_unconverged = 0

    def update(self, dt, T, rho, dust_frac, chem, grain_cross_section=None,
               **kwargs):
        """Update the gas/ice abundances"""
        g = self._grains(dust_frac, chem.ice, grain_cross_section)
        self.n_unconverged = 0
        for spec in chem:
            if 'grain' in spec:
                self._update_ice_balance(dt, T, rho, dust_frac, spec, chem)
            else:
                self._update_shell(dt, T, rho, g, spec, chem)

    def _grains(self, dust_frac, ice, grain_cross_section=None):
        """The grains without their shells: dust fraction ('base'), the
        effective dust fraction of ThermalChem ('eps'), the power law (s, L,
        amin) and its mass-weighted moments R_j = <a^-j>, j = 0..3"""
        if grain_cross_section is None:
            grain_cross_section = self._disc.grain_cross_section
        shells = sum(np.maximum(ice.area(s), 0) for s in ice.area_species)
        base = np.maximum(dust_frac - shells, 0)
        _, s, L = self._disc.size_distribution()
        amin = self._disc.amin
        R = np.array([moment_ratio(-j, s, L, amin) for j in range(4)])
        return dict(base=base, eps=self._effective_dust_frac(base, grain_cross_section),
                    s=s, L=L, amin=amin, R=R)

    def thickness(self, dust_frac, ice, spec, grain_cross_section=None):
        """Thickness of spec's shell [cm]"""
        g = self._grains(dust_frac, ice, grain_cross_section)
        ok = g['base'] > 0
        r = np.where(ok, np.maximum(ice.area(spec), 0)/np.where(ok, g['base'], 1), 0.)
        return shell_thickness(r, g['s'], g['L'], g['amin'])

    def _update_shell(self, dt, T, rho, g, spec, chem):
        ice, gas = chem.ice, chem.gas
        if spec not in ice.area_species:
            raise KeyError("{} has no shell".format(spec))

        m_mol, mu = ice.mass(spec), self._mu
        conv = mu/m_mol
        base = g['base']
        A0, M0 = ice.area(spec), ice.mass_part(spec)
        X_t = (A0 + M0 + gas[spec])*conv

        n = rho/(mu*m_H)
        Tbind = self._Tbind[spec]
        Sa = self._f_ads*self._v_therm(T, m_mol)*g['eps']*n
        Sd = self._f_des*self._nu_i(Tbind, m_mol)*np.exp(-Tbind/T)*np.ones_like(X_t)
        Xm = self._etaNbind*g['eps']        # a monolayer on the grains

        ok = (Xm > 0) & (base > 0) & (X_t > 0)
        if not ok.any():
            return
        k = np.nonzero(ok)[0]
        b, s, L, amin, R = base[k], g['s'][k], g['L'][k], g['amin'], g['R'][:, k]
        Xm, Sd, X_t = Xm[k], Sd[k], X_t[k]
        ra = Sa[k]/Xm                           # layers per unit time, per X_g
        # Monolayer thickness: its mass over the grains' area times density
        delta1 = Xm/(conv*3*b*R[1])

        # The grains: core c = kappa a (the volatile's mass-distributed ice
        # uniform by mass), coated radius a + d
        A, M = np.maximum(A0[k], 0), np.maximum(M0[k], 0)
        v = np.clip(M/b, 0, 1)
        kappa = np.cbrt(1 - v)
        beta = v/(1 + kappa*(1 + kappa))       # 1 - kappa
        d = shell_thickness(A/b, s, L, amin)
        loss = dt*Sd

        # The fractions of the mass moments R_j below each size, and the
        # ice per unit grain mass with t = p + q a on [x1, x2]:
        # ((kappa + q) a + p)^3/a^3 - kappa^3, and its derivative in p
        def moments_below(x, i):
            return np.array([R[j, i]*_frac_below(s[i] - j, x, L[i]) for j in range(4)])

        def segment(p, q, dP, i):
            kp = kappa[i]
            K = kp + q
            ice = q*(K*K + K*kp + kp*kp)*dP[0] + p*(3*K*K*dP[1] + p*(3*K*dP[2] + p*dP[3]))
            dice = 3*K*K*dP[1] + p*(6*K*dP[2] + 3*p*dP[3])
            return ice, dice

        def ice_of(Xg, i):
            """The ice per unit grain mass at the end of the step, for the
            vapour Xg at its end, and its derivative in Xg (cells i)"""
            dl, l, bt, dd = delta1[i], loss[i], beta[i], d[i]
            gain_rate = dt*ra[i]
            gain = gain_rate*Xg
            # Grains with t >= t_star at the start end above a monolayer
            t_star = dl*(1 + l - gain)
            with np.errstate(divide='ignore', invalid='ignore'):
                am = np.where(bt > 0, (t_star - dd)/bt, np.where(dd >= t_star, 0., np.inf))
                xm = np.clip(np.log(np.maximum(am, 1e-300)/amin), 0, L[i])
            mid = moments_below(xm, i)
            sub, dsub = segment((dd + dl*gain)/(1 + l), bt/(1 + l), mid, i)
            multi, dmulti = segment(dd + dl*(gain - l), bt, R[:, i] - mid, i)
            return sub + multi, dl*gain_rate*(dsub/(1 + l) + dmulti)

        def fdf(Xg, i):
            I, dI = ice_of(Xg, i)
            c = conv*b[i]
            return Xg + c*I - X_t[i], 1 + c*dI

        tol = self._tol*X_t
        Xg0 = np.clip(gas[spec][k]*conv, 0, X_t)
        Xg, conv_k = _newton_root(fdf, Xg0, np.zeros_like(X_t), X_t.copy(), tol)
        self.n_unconverged += int((~conv_k).sum())

        # Conserving the total: the ice at the root, unless the ice is too
        # steep in the vapour for that (long steps), then the root's vapour
        all_k = np.arange(len(k))
        I1 = b*ice_of(Xg, all_k)[0]
        steep = np.abs(Xg + conv*I1 - X_t) > tol
        I1 = np.clip(np.where(steep, (X_t - Xg)/conv, I1), 0, X_t/conv)
        Xg = X_t - conv*I1

        # Gains to the shell, losses from it first; the equilibrium coverage
        # below a monolayer, on the cores (area kappa^2), to the shell
        with np.errstate(divide='ignore', invalid='ignore'):
            theta = np.where(Sd > 0, np.minimum(ra*Xg/Sd, 1), 1.)
        floor = theta*kappa*kappa*Xm/conv
        M1 = np.minimum(M, np.maximum(I1 - floor, 0))

        A_new, M_new, g_new = A0.copy(), M0.copy(), gas[spec].copy()
        A_new[k], M_new[k], g_new[k] = I1 - M1, M1, Xg/conv
        ice.set_parts(spec, A_new, M_new)
        gas[spec] = g_new
