# area_ice.py
#
# Thermal adsorption/desorption with the ice of each volatile split into a
# part distributed by grain surface area and a part distributed by grain mass,
# for the TriMom dust model (dust_trimom.DustGrowthTriMom).
#
# Per grain of size a the ice coverage, in monolayers, is
#     theta(a) = (A + M a / a_A) / X_mono,
# with A and M the area- and mass-distributed ice, a_A = M_0/M_-1 the
# area-weighted mean size and X_mono the monolayer capacity of the dust
# (ThermalChem's X_max with the dust's own area). The area part is a mantle of
# uniform thickness; the mass part has thickness ~ a.
#
# Per grain, adsorption adds coverage uniformly (rate ~ X_gas) and desorption
# removes it at k_d min(theta, 1) (first order below a monolayer, zeroth order
# above, as ThermalChem). Projected onto (A, M), with X = A + M the ice:
#     dX/dt = Sa (X_t - X) - Sd [A P_A + M P_M + X_mono (1 - P_A)]
#     dM/dt = - Sd M P_M
# where P_A (P_M) is the fraction of the area (of the mass-distributed ice)
# on grains below a monolayer, a < a_c = a_A (X_mono - A)/M. Below a monolayer
# molecules desorb at Sd and re-adsorb by area, which moves the mass-
# distributed ice into the area-distributed ice. Above, adsorption and
# desorption both add or remove uniform layers: M is unchanged.
#
# Transport (MomentDrift(area_tracers=...)) moves A with the area-weighted
# velocity and M with the mass-weighted one, and growth (DustGrowthTriMom)
# mixes A into M (mix_area_to_mass).
################################################################################
import numpy as np

from ..constants import m_H
from .base_chem import ChemicalAbund, ThermalChem, SimpleChemBase


class AreaMassIceAbund(ChemicalAbund):
    """Ice abundances with the ice of the volatiles in area_species split
    into an area-distributed part (rows '<spec>:A') and a mass-distributed
    part (rows '<spec>'). The other species (e.g. refractory grains) are
    distributed by mass.

    ice[spec] is the total; setting it keeps the area part (up to the new
    total). data holds all the rows, which the driver transports; area_rows
    are the indices to pass to MomentDrift(area_tracers=...).

    args:
        species      : list of species names
        masses       : their molecular masses in amu
        area_species : the species with an area-distributed part
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
        """Indices of the area-distributed rows in data"""
        return np.array([self._indexes[s + ':A'] for s in self._area_species])

    def area(self, k):
        """Area-distributed ice of species k"""
        return self._data[self._indexes[k + ':A']]

    def mass_part(self, k):
        """Mass-distributed ice of species k"""
        return self._data[self._indexes[k]]

    def set_parts(self, k, area, mass):
        """Set the area- and mass-distributed ice of species k"""
        self._data[self._indexes[k + ':A']] = area
        self._data[self._indexes[k]] = mass

    def mix_area_to_mass(self, f):
        """Move a fraction f of the area-distributed ice into the mass-
        distributed ice (growth mixing)"""
        iA = self.area_rows
        iM = np.array([self._indexes[s] for s in self._area_species])
        dA = f*self._data[iA]
        self._data[iM] += dA
        self._data[iA] -= dA


def _frac_below(k, xi, L):
    """int_0^xi e^(k x) dx / int_0^L e^(k x) dx, for 0 <= xi <= L"""
    small = np.abs(k*L) < 1e-8
    ks = np.where(small, 1., k)
    return np.where(small, xi/L, np.expm1(ks*xi)/np.expm1(ks*L))


def _h(z):
    """(1 - e^-z)/z, = 1 at z = 0 (z >= 0)"""
    small = z < 1e-8
    zs = np.where(small, 1., z)
    return np.where(small, 1 - 0.5*z, -np.expm1(-zs)/zs)


def _xi_of(p, X, M):
    """ln(a_c/amin) of the state (X, M), in [0, L]"""
    L = p['L']
    with np.errstate(divide='ignore', invalid='ignore'):
        x = p['lr'] + np.log((p['Xm'] - (X - M))/M)
    x = np.where(M > 0, x, np.where(X - M < p['Xm'], L, 0.))
    return np.clip(np.nan_to_num(x, nan=0., posinf=L, neginf=0.), 0, L)


def _solve(p, X, M, xi, t):
    """(X, M) after a time t with the coefficients at xi"""
    L, s = p['L'], p['s']
    PA = _frac_below(s - 1, xi, L)
    PM = _frac_below(s, xi, L)
    lam = p['Sa'] + p['Sd']*PA
    gam = p['Sd']*PM
    c = p['Sa']*p['X_t'] - p['Sd']*p['Xm']*(1 - PA)
    beta = p['Sd']*(PM - PA)
    X1 = X*np.exp(-lam*t) + c*t*_h(lam*t) - \
        beta*M*t*np.exp(-gam*t)*_h((lam - gam)*t)
    return X1, M*np.exp(-gam*t)


class TimeDependentChemTriMom(ThermalChem, SimpleChemBase):
    """Time dependent thermal adsorption/desorption with area- and mass-
    distributed ice (AreaMassIceAbund), for the TriMom dust model.

    Uses the size distribution of disc (a DustGrowthTriMom) and the rates of
    ThermalChem with the dust's own grain area. The chemistry is a separate
    operator: transport moves the two parts with their own velocities and
    growth mixes them (MomentDrift(area_tracers=...), DustGrowthTriMom).

    The step is exact within each regime (all grains below a monolayer, or
    all above) and finds the time at which the regime changes. The mixed
    regime is integrated in sub-steps, each exact for the coefficients of its
    start, that change ln(a_c) by at most dxi (down to 1/4096 of the step).
    With no mass-distributed ice it reduces to TimeDependentChem.

    The area-distributed ice can strictly go a little negative (a uniform
    layer removed by desorption can leave the smallest grains bare, down to
    A = -M a_min/a_A); it is kept >= 0 by moving any deficit into the mass-
    distributed ice. last_clip holds the mass abundance moved in the last
    step, per species, and n_unconverged the number of cells whose mixed-
    regime integration reached max_substeps.

    args:
        disc         : the DustGrowthTriMom disc
        dxi          : largest change of ln(a_c) per sub-step in the mixed
                       regime, default 0.05
        max_substeps : most sub-steps in the mixed regime, default 1000
        other args as ThermalChem
    """
    def __init__(self, disc, dxi=0.05, max_substeps=1000, **kwargs):
        ThermalChem.__init__(self, **kwargs)
        SimpleChemBase.__init__(self)
        self._disc = disc
        self._dxi = dxi
        self._max_substeps = max_substeps
        self.last_clip = {}
        self.n_unconverged = 0

    def update(self, dt, T, rho, dust_frac, chem, grain_cross_section=None,
               **kwargs):
        """Update the gas/ice abundances"""
        if grain_cross_section is None:
            grain_cross_section = self._disc.grain_cross_section
        eps = self._effective_dust_frac(dust_frac, grain_cross_section)
        self.n_unconverged = 0
        for spec in chem:
            if 'grain' in spec:
                self._update_ice_balance(dt, T, rho, dust_frac, spec, chem)
            else:
                self._update_two_parts(dt, T, rho, eps, spec, chem)

    def _update_two_parts(self, dt, T, rho, eps, spec, chem):
        ice, gas = chem.ice, chem.gas
        if spec not in ice.area_species:
            raise KeyError("{} has no area-distributed ice".format(spec))
        m_mol, mu = ice.mass(spec), self._mu
        conv = mu/m_mol

        A0 = ice.area(spec)*conv
        M0 = ice.mass_part(spec)*conv
        X_t = A0 + M0 + gas[spec]*conv
        X0 = A0 + M0

        n = rho/(mu*m_H)
        Tbind = self._Tbind[spec]
        Sa = self._f_ads*self._v_therm(T, m_mol)*eps*n
        Sd = self._f_des*self._nu_i(Tbind, m_mol)*np.exp(-Tbind/T)
        Xm = self._etaNbind*eps

        _, s, L = self._disc.size_distribution()
        lr = self._ln_aA(s, L)            # ln(a_A/amin)

        P = dict(Sa=Sa, Sd=Sd, Xm=Xm, X_t=X_t, s=s, L=L, lr=lr)

        # Regimes: all below a monolayer (xi = L), all above (xi = 0)
        def g_sub(p, X, M):               # <= 0: all below a monolayer
            return (X - M) + M*np.exp(p['L'] - p['lr']) - p['Xm']
        def g_multi(p, X, M):             # >= 0: all above
            return (X - M) + M*np.exp(-p['lr']) - p['Xm']

        X, M = X0.copy(), M0.copy()
        t_left = np.full_like(X, dt)
        sub = g_sub(P, X, M) <= 0
        multi = (g_multi(P, X, M) >= 0) & ~sub
        pure = sub | multi
        xi_p = np.where(sub, L, 0.)

        # Pure regimes: exact, unless the regime is left during the step
        X1, M1 = _solve(P, X, M, xi_p, t_left)
        stay = np.where(sub, g_sub(P, X1, M1) <= 0, g_multi(P, X1, M1) >= 0) & pure
        cross = pure & ~stay
        if cross.any():
            # Time at which the regime is left (bisection)
            lo, hi = np.zeros_like(X), t_left.copy()
            def g(Xa, Ma):
                return np.where(sub, g_sub(P, Xa, Ma), -g_multi(P, Xa, Ma))
            for _ in range(60):
                mid = 0.5*(lo + hi)
                Xa, Ma = _solve(P, X, M, xi_p, mid)
                out = g(Xa, Ma) > 0
                hi = np.where(out, mid, hi)
                lo = np.where(out, lo, mid)
            Xc, Mc = _solve(P, X, M, xi_p, lo)
            X = np.where(cross, Xc, X)
            M = np.where(cross, Mc, M)
            t_left = np.where(cross, t_left - lo, t_left)

        X_end = np.where(stay, X1, X)
        M_end = np.where(stay, M1, M)

        # Uniform ice (M = 0) that crossed: the other regime for the rest of
        # the step (as TimeDependentChem)
        flip = cross & (M <= 0)
        if flip.any():
            Xa, Ma = _solve(P, X, M, np.where(sub, 0., L), t_left)
            X_end = np.where(flip, Xa, X_end)
            M_end = np.where(flip, Ma, M_end)

        # Mixed regime (and after a change of regime): sub-steps with the
        # coefficients of the sub-step's start, each exact for them, limited
        # to a change of xi of DXI (down to a minimum sub-step)
        mixed = ~stay & ~flip
        if mixed.any():
            k = np.nonzero(mixed)[0]
            p = {key: v[k] for key, v in P.items()}
            Xk, Mk, tl = X[k], M[k], t_left[k]
            h_min = tl/4096.
            h = tl/16.
            from_sub = (cross & sub)[k]
            for it in range(self._max_substeps):
                act = tl > 0
                if not act.any():
                    break
                hh = np.minimum(h, tl)
                xi_s = _xi_of(p, Xk, Mk)
                if it == 0:
                    # at the boundary just crossed
                    xi_s = np.where(cross[k], np.where(from_sub, p['L'], 0.), xi_s)
                Xn, Mn = _solve(p, Xk, Mk, xi_s, hh)
                ok = act & ((np.abs(_xi_of(p, Xn, Mn) - xi_s) <= self._dxi) |
                            (hh <= h_min))
                Xk = np.where(ok, Xn, Xk)
                Mk = np.where(ok, Mn, Mk)
                tl = np.where(ok, tl - hh, tl)
                h = np.where(ok, 1.5*hh, 0.5*hh)
            else:
                # Finish any cells left in one sub-step
                act = tl > 0
                Xn, Mn = _solve(p, Xk, Mk, _xi_of(p, Xk, Mk), tl)
                Xk = np.where(act, Xn, Xk)
                Mk = np.where(act, Mn, Mk)
                self.n_unconverged += int(act.sum())
            X_end[k], M_end[k] = Xk, Mk

        # Keep 0 <= X <= X_t and A >= 0 (moving any deficit of A into M)
        X_end = np.clip(X_end, 0, X_t)
        M_end = np.clip(M_end, 0, None)
        A_end = X_end - M_end
        deficit = np.maximum(-A_end, 0)
        M_end -= deficit
        A_end += deficit
        self.last_clip[spec] = deficit/conv

        ice.set_parts(spec, A_end/conv, M_end/conv)
        gas[spec] = np.maximum(X_t - X_end, 0)/conv

    @staticmethod
    def _ln_aA(s, L):
        """ln(a_A/amin): a_A/amin = int_0^L e^(s x) dx / int_0^L e^((s-1) x) dx"""
        def lnG(k):
            small = np.abs(k*L) < 1e-8
            ks = np.where(small, 1., k)
            # ln((e^(kL) - 1)/k), stable for either sign of k
            big = np.where(small, np.log(L), np.log(np.abs(np.expm1(ks*L))) - np.log(np.abs(ks)))
            return big
        return lnG(s) - lnG(s - 1)
