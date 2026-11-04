# area_ice.py
#
# Thermal adsorption/desorption with the ice of each volatile split into a
# part distributed by grain surface area and a part distributed by grain mass,
# for the TriMom dust model (dust_trimom.DustGrowthTriMom).
#
# Per grain, adsorption adds coverage (in monolayers) uniformly, at a rate
# ~ X_gas, and desorption removes it at k_d min(theta, 1): first order below a
# monolayer, zeroth order above, as ThermalChem. Every grain follows the same
# equation, so at a sublimation front the small grains empty first and the
# large ones keep their ice for longer.
#
# The ice of a volatile on a grain of size a is represented by the profile
#     theta(a) = max(phi, (alpha + beta a/a_A)/X_mono)
# with a_A = M_0/M_-1 the area-weighted mean size and X_mono the monolayer
# capacity of the dust (ThermalChem's X_max with the dust's own area): a
# floor phi on the small grains (bare or below a monolayer) and a line, a
# uniform mantle plus one of thickness ~ a. Two rows are stored, A and M, read
# as moments of the ice: X = A + M and its a/a_A-weighted moment
# Y = A + c M (c = <a^2>_area/a_A^2). Without a floor region they are the
# line's coefficients. Each step takes the floor in equilibrium with the gas
# (out of A), rebuilds the line from (X, Y), and then either solves a pure
# regime exactly or takes one step implicit in the gas, the line shifted with
# its slope fixed and the floor integrated (two_component_ice.tex in
# scripts/trimom/production/ice_area/).
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

    def clip_negative(self):
        """Set negative abundances to zero, except the area-distributed ice,
        which can be negative (the rows are the moments X = A + M and
        Y = A + c M of a profile that increases with size, so the realisable
        rows are X >= 0 and 0 <= M <= X (amax/a_A - 1)/(c - 1); the upper
        bound needs the size distribution and is kept by
        TimeDependentChemTriMom). Negative M is moved into A, keeping X
        (uniform ice, Y = X); a negative total is set to zero, with both
        rows (X = 0 needs Y = 0)."""
        iA = self.area_rows
        iM = np.array([self._indexes[s] for s in self._area_species])
        rest = np.setdiff1d(np.arange(len(self._data)), np.concatenate([iA, iM]))
        self._data[rest] = np.maximum(self._data[rest], 0)
        A, M = self._data[iA], self._data[iM]
        neg = np.minimum(M, 0)
        M -= neg
        A += neg
        empty = A + M < 0
        self._data[iA] = np.where(empty, 0., A)
        self._data[iM] = np.where(empty, 0., M)


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


PHI_MARGIN = 1e-3      # margin from a delta at amax (floor and realisability)


def _lnG(k, L):
    """ln G_k(L) = ln((e^(kL) - 1)/k), stable for either sign of k"""
    small = np.abs(k*L) < 1e-8
    ks = np.where(small, 1., k)
    return np.where(small, np.log(L),
                    np.log(np.abs(np.expm1(ks*L))) - np.log(np.abs(ks)))


def _shape(s, L):
    """The size distribution as the ice profile sees it: s, L, lr =
    ln(a_A/amin), c = <a^2>_area/a_A^2, amin_r = amin/a_A, amax_r = amax/a_A"""
    lr = _lnG(s, L) - _lnG(s - 1, L)
    c = np.exp(_lnG(s + 1, L) - _lnG(s - 1, L) - 2*lr)
    return dict(s=s, L=L, lr=lr, c=c, amin_r=np.exp(-lr), amax_r=np.exp(L - lr))


def _fracs(g, x):
    """(P_A, P_M, P_C): the fractions of the area and of its a- and
    a^2-moments on grains below x = ln(a/amin)"""
    s, L = g['s'], g['L']
    return (_frac_below(s - 1, x, L), _frac_below(s, x, L),
            _frac_below(s + 1, x, L))


def _moments(g, Xm, phi, alpha, beta, xf):
    """(X, Y) = (int n_i, int (a/a_A) n_i) of the profile
    max(phi Xm, alpha + beta a/a_A), with the floor below xf"""
    PA, PM, PC = _fracs(g, xf)
    X = Xm*phi*PA + alpha*(1 - PA) + beta*(1 - PM)
    Y = Xm*phi*PM + alpha*(1 - PM) + beta*g['c']*(1 - PC)
    return X, Y


def _cutoff_moments(g, x):
    """(I_1, I_2): the moments of (a - a_f)_+/a_A, with a_f = amin e^x"""
    PA, PM, PC = _fracs(g, x)
    af = g['amin_r']*np.exp(x)
    return (1 - PM) - af*(1 - PA), g['c']*(1 - PC) - af*(1 - PM)


def _sub(g, k):
    """The values of g in the cells k"""
    return {key: v[k] for key, v in g.items()}


def _profile(g, Xm, X, Y, phi):
    """(alpha, beta, xf): the line of the profile max(phi Xm, alpha + beta
    a/a_A) with moments (X, Y). The floor is taken out of the area part,
    X' = X - phi Xm, Y' = Y - phi Xm; the rest is the line (A', M) if it is
    above the floor at amin, otherwise a line cut off at a_f > amin, with
    I_2/I_1(xf) = Y'/X' (monotone in xf)"""
    Xp, Yp = X - phi*Xm, Y - phi*Xm
    beta = (Yp - Xp)/(g['c'] - 1)
    alpha = Xp - beta
    xf = np.zeros_like(X)
    cut = (alpha < -beta*g['amin_r']) & (Xp > 0)
    if cut.any():
        k = np.nonzero(cut)[0]
        gk = _sub(g, k)
        target = Yp[k]/Xp[k]
        lo, hi = np.zeros(len(k)), gk['L'].copy()
        for _ in range(60):
            mid = 0.5*(lo + hi)
            I1, I2 = _cutoff_moments(gk, mid)
            above = I2 > target*I1
            hi = np.where(above, mid, hi)
            lo = np.where(above, lo, mid)
        x = 0.5*(lo + hi)
        I1, _ = _cutoff_moments(gk, x)
        beta[k] = Xp[k]/np.maximum(I1, 1e-300)
        alpha[k] = -beta[k]*gk['amin_r']*np.exp(x)
        xf[k] = x
    return alpha + phi*Xm, beta, xf


def _implicit_step(g, Sa, Sd, Xm, X_t, X0, alpha0, beta, phi_s, dt,
                   tol=1e-12, max_iter=200):
    """One step of the profile max(phi Xm, alpha + beta a/a_A), implicit in
    the gas: with the adsorption rate r_1 = Sa (X_t - X_1)/Xm of the end
    state,
        alpha_1 = alpha_0 + Xm D,  D = dt (r_1 - Sd)  (the line: every grain
                                                       above a monolayer
                                                       shifts by D layers)
        phi_1   = phi_s + dt [r_1 - Sd min(phi_1, 1)]  (the floor)
        X_1     = X(phi_1, alpha_1, beta)              (moments of the profile)
    beta fixed. Line grains that empty within the step end on the floor
    (the max), so the stripping at a front is exact however far the cut-off
    moves. Solved for D (well scaled however long the step): the residual
    X_1(D) - X(phi_1(D), alpha_1(D), beta) decreases with D, one root.

    returns X_1, phi_1, alpha_1, xf_1, xc_1 and the converged mask"""
    L, amin_r = g['L'], g['amin_r']

    def state(D):
        X1 = X_t - Xm*(Sd + D/dt)/Sa
        phi1 = (phi_s + D + Sd*dt)/(1 + Sd*dt)
        multi = phi1 >= 1
        phi1 = np.where(multi, phi_s + D, phi1)
        alpha = alpha0 + Xm*D
        xf = np.log(np.maximum((phi1*Xm - alpha)/(beta*amin_r), 1e-300))
        xf = np.clip(xf, 0, L)
        xc = np.log(np.maximum((Xm - alpha)/(beta*amin_r), 1e-300))
        xc = np.where(multi, xf, np.clip(xc, xf, L))
        X, _ = _moments(g, Xm, phi1, alpha, beta, xf)
        return X1 - X, phi1, alpha, xf, xc

    # D from r_1 = 0 (all gas adsorbed... none: X_1 = X_t) to X_1 = 0
    a = -Sd*dt
    b = dt*(Sa*X_t/Xm - Sd)
    fa, fb = state(a)[0], state(b)[0]
    z = a.copy()
    side = np.zeros(len(a), dtype=int)
    done = fa <= 0                      # the end state takes all the gas
    width = b - a
    bisect = np.zeros(len(a), dtype=bool)
    scale = tol*np.maximum(X_t, 1e-300)
    for _ in range(max_iter):
        # Illinois (modified regula falsi); bisection when it stalls
        with np.errstate(divide='ignore', invalid='ignore'):
            zn = (a*fb - b*fa)/(fb - fa)
        ok = np.isfinite(zn) & (zn > a) & (zn < b) & ~bisect
        zn = np.where(ok, zn, 0.5*(a + b))
        z = np.where(done, z, zn)
        fz = state(z)[0]
        pos = fz > 0
        a = np.where(pos & ~done, z, a)
        fa = np.where(pos & ~done, fz, fa)
        b = np.where(~pos & ~done, z, b)
        fb = np.where(~pos & ~done, fz, fb)
        fb = np.where(pos & (side == 1), 0.5*fb, fb)
        fa = np.where(~pos & (side == -1), 0.5*fa, fa)
        side = np.where(pos, 1, -1)
        bisect = (b - a) > 0.5*width
        width = b - a
        done |= (np.abs(fz) <= scale) | (b - a <= 1e-15*(np.abs(a) + np.abs(b)))
        if done.all():
            break
    _, phi1, alpha, xf, xc = state(z)
    X1, _ = _moments(g, Xm, phi1, alpha, beta, xf)
    return X1, phi1, alpha, xf, xc, done


def _realisable(X, M, g):
    """(M, amount moved): the ice can be no more concentrated on the largest
    grains than all of it at amax, Y/X <= amax/a_A, i.e.
    M <= X (amax/a_A - 1)/(c - 1), less the margin PHI_MARGIN (at the bound
    the profile is a delta at amax); beyond it mass-distributed ice is moved
    into the area part, keeping X"""
    M_max = (1 - PHI_MARGIN)*X*(g['amax_r'] - 1)/(g['c'] - 1)
    M_new = np.minimum(M, np.maximum(M_max, 0))
    return M_new, M - M_new


def _pure(sub, X, M, Sa, Sd, Xm, X_t, t):
    """(X, M) after a time t in a pure regime, exact for any profile:
    all grains above a monolayer (a uniform shift),
        dX/dt = Sa (X_t - X) - Sd Xm,   M fixed;
    all below (sub; one linear map of every grain's coverage),
        dX/dt = Sa (X_t - X) - Sd X,    dM/dt = -Sd M"""
    lam = np.where(sub, Sa + Sd, Sa)
    c = np.where(sub, Sa*X_t, Sa*X_t - Sd*Xm)
    X1 = X*np.exp(-lam*t) + c*t*_h(lam*t)
    return X1, np.where(sub, M*np.exp(-Sd*t), M)


class TimeDependentChemTriMom(ThermalChem, SimpleChemBase):
    """Time dependent thermal adsorption/desorption with area- and mass-
    distributed ice (AreaMassIceAbund), for the TriMom dust model.

    Uses the size distribution of disc (a DustGrowthTriMom) and the rates of
    ThermalChem with the dust's own grain area. The chemistry is a separate
    operator: transport moves the two parts with their own velocities and
    growth mixes them (MomentDrift(area_tracers=...), DustGrowthTriMom).

    The rows are read as moments of the ice: X = A + M and its a/a_A-weighted
    moment Y = A + c M. Each step rebuilds the profile
        theta(a) = max(phi, (alpha + beta a/a_A)/X_mono)
    with the floor phi (the small, sub-monolayer or bare grains) in
    equilibrium with the gas, taken out of the area part, and the rest a
    line, cut off at a_f if it would fall below the floor
    (notes/bare_grains.md). Cells that stay in a pure regime (all grains
    above a monolayer, or all below) take the exact solution, and uniform
    ice (M = 0) reduces to TimeDependentChem. The others take one step
    implicit in the gas: every line grain above a monolayer shifts by the
    same, the floor is integrated, beta is fixed, and a_f, a_c are those of
    the end state (two_component_ice.tex).

    last_clip holds the mass abundance moved by the realisability bound in
    the last step (Y/X <= amax/a_A), per species; last_floor the floor ice at
    the start and the end of the step (and phi), the ice on the band
    (between the floor and one layer) at the end and the cells that took the
    implicit step ('mixed'), per species; n_unconverged the number of
    cells whose implicit step did not converge.

    args:
        disc : the DustGrowthTriMom disc
        tol  : relative tolerance of the implicit step, default 1e-12
        other args as ThermalChem
    """
    def __init__(self, disc, tol=1e-12, **kwargs):
        ThermalChem.__init__(self, **kwargs)
        SimpleChemBase.__init__(self)
        self._disc = disc
        self._tol = tol
        self.last_clip = {}
        self.last_floor = {}
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

        b = self._rebuild(T, rho, eps, spec, chem)
        conv = b['conv']
        g, X_t, X0, M0, Y0 = b['g'], b['X_t'], b['X'], b['M'], b['Y']
        Sa, Sd, Xm, ok = b['Sa'], b['Sd'], b['Xm'], b['ok']
        phi_s, alpha, beta, xf = b['phi'], b['alpha'], b['beta'], b['xf']
        moved0 = b['moved']
        amin_r, amax_r = g['amin_r'], g['amax_r']
        Xm_ = np.where(ok, Xm, 1.)

        # Regimes: all grains below a monolayer, or all above
        th_min = np.maximum(phi_s, (alpha + beta*amin_r)/Xm_)
        th_max = np.maximum(phi_s, (alpha + beta*amax_r)/Xm_)
        sub = (th_max <= 1) & ok
        multi = (th_min >= 1) & ~sub & ok
        pure = sub | multi

        # Pure regimes: exact, unless the regime is left during the step
        X1, M1 = _pure(sub, X0, M0, Sa, Sd, Xm, X_t, dt)
        e = np.exp(-Sd*dt)
        stay = np.where(sub, th_max*e + (X1 - X0*e)/Xm_ <= 1,
                        th_min + (X1 - X0)/Xm_ >= 1) & pure
        X_end = np.where(stay, X1, X0)
        M_end = np.where(stay, M1, M0)
        floor_end = np.where(sub, phi_s*e + (X1 - X0*e)/Xm_, phi_s + (X1 - X0)/Xm_)
        PA_s = _fracs(g, xf)[0]
        floor0 = Xm*phi_s*PA_s
        floor1 = np.where(stay, Xm*floor_end*PA_s, floor0)
        phi_end = np.where(stay, floor_end, phi_s)
        band1 = np.zeros_like(X0)

        # Uniform ice (M = 0) that leaves its regime: the crossing time, then
        # the other regime for the rest of the step (as TimeDependentChem)
        uni = pure & ~stay & (M0 <= 0)
        if uni.any():
            k = np.nonzero(uni)[0]
            Xk, Mk, sk = X0[k], M0[k], sub[k]
            r = (Sa[k], Sd[k], Xm[k], X_t[k])
            lo, hi = np.zeros(len(k)), np.full(len(k), dt)
            for _ in range(60):
                mid = 0.5*(lo + hi)
                Xa, _ = _pure(sk, Xk, Mk, *r, mid)
                out = np.where(sk, Xa > Xm[k], Xa < Xm[k])
                hi = np.where(out, mid, hi)
                lo = np.where(out, lo, mid)
            Xc, Mc = _pure(sk, Xk, Mk, *r, lo)
            X_end[k], M_end[k] = _pure(~sk, Xc, Mc, *r, dt - lo)

        # Everything else (the mixed regime): one step implicit in the gas
        mixed = ~stay & ~uni & ok
        if mixed.any():
            k = np.nonzero(mixed)[0]
            gk = _sub(g, k)
            X1k, phi1, alpha1, xf1, xc1, conv_k = _implicit_step(
                gk, Sa[k], Sd[k], Xm[k], X_t[k], X0[k], alpha[k], beta[k],
                phi_s[k], dt, tol=self._tol)
            Xk, Yk = _moments(gk, Xm[k], phi1, alpha1, beta[k], xf1)
            X_end[k] = Xk
            M_end[k] = (Yk - Xk)/(gk['c'] - 1)
            PAf, PMf, _ = _fracs(gk, xf1)
            PAc, PMc, _ = _fracs(gk, xc1)
            floor1[k] = Xm[k]*np.minimum(phi1, 1)*PAf
            phi_end[k] = phi1
            band1[k] = np.where(phi1 < 1, alpha1*(PAc - PAf) + beta[k]*(PMc - PMf), 0.)
            self.n_unconverged += int((~conv_k).sum())

        # Keep 0 <= X <= X_t, M >= 0 and Y/X <= amax/a_A
        X_end = np.clip(X_end, 0, X_t)
        M_end = np.clip(M_end, 0, None)
        M_end, moved = _realisable(X_end, M_end, g)
        A_end = X_end - M_end
        self.last_clip[spec] = (moved0 + moved)/conv
        self.last_floor[spec] = dict(start=floor0/conv, end=floor1/conv,
                                     band=band1/conv, mixed=mixed,
                                     phi_start=phi_s, phi_end=phi_end)

        ice.set_parts(spec, A_end/conv, M_end/conv)
        gas[spec] = np.maximum(X_t - X_end, 0)/conv

    def _rebuild(self, T, rho, eps, spec, chem):
        """The start of a step: rates, the realisable rows and the profile
        max(phi, (alpha + beta a/a_A)/Xm), with the floor phi in equilibrium
        with the gas (at most one layer, and no more than leaves the rest
        realisable), taken out of the area part. Abundances in units of the
        number abundance per H (conv)."""
        ice, gas = chem.ice, chem.gas
        m_mol, mu = ice.mass(spec), self._mu
        conv = mu/m_mol

        A0 = ice.area(spec)*conv
        M0 = ice.mass_part(spec)*conv
        X_t = A0 + M0 + gas[spec]*conv
        X0 = A0 + M0

        _, s, L = self._disc.size_distribution()
        g = _shape(s, L)

        # Transport moves A and M separately: start from a realisable state
        M0, moved = _realisable(X0, M0, g)
        Y0 = X0 + (g['c'] - 1)*M0

        n = rho/(mu*m_H)
        Tbind = self._Tbind[spec]
        Sa = self._f_ads*self._v_therm(T, m_mol)*eps*n
        Sd = self._f_des*self._nu_i(Tbind, m_mol)*np.exp(-Tbind/T)
        Xm = self._etaNbind*eps*np.ones_like(X0)
        ok = Xm > 0
        Xm_ = np.where(ok, Xm, 1.)

        X_g = X_t - X0
        with np.errstate(divide='ignore', invalid='ignore'):
            theta_eq = np.where(X_g > 0, Sa*X_g/(Sd*Xm_), 0.)
        theta_eq = np.nan_to_num(theta_eq, nan=0., posinf=1.)
        # At phi_max the rest would be all at amax (a delta, beta = inf): keep
        # a margin, which leaves it on a narrow band of sizes below amax
        phi_max = (1 - PHI_MARGIN)*(X0 - Y0/g['amax_r'])/(Xm_*(1 - 1/g['amax_r']))
        phi = np.where(ok, np.clip(np.minimum(np.minimum(theta_eq, 1), phi_max), 0, 1), 0.)
        alpha, beta, xf = _profile(g, Xm, X0, Y0, phi)
        return dict(g=g, conv=conv, X_t=X_t, X=X0, M=M0, Y=Y0, moved=moved,
                    Sa=Sa, Sd=Sd, Xm=Xm, ok=ok, phi=phi, alpha=alpha,
                    beta=beta, xf=xf)

    def ice_profile(self, T, rho, dust_frac, chem, spec,
                    grain_cross_section=None):
        """The profile of spec's ice as a step would rebuild it,
        theta(a) = max(phi, (alpha + beta a/a_A)/Xm): a dict with phi, alpha,
        beta and Xm (number abundance per H), xf = ln(a_f/amin) and the shape
        g (s, L, lr, c, amin_r, amax_r)"""
        if grain_cross_section is None:
            grain_cross_section = self._disc.grain_cross_section
        eps = self._effective_dust_frac(dust_frac, grain_cross_section)
        b = self._rebuild(T, rho, eps, spec, chem)
        return {k: b[k] for k in ('phi', 'alpha', 'beta', 'xf', 'Xm', 'g')}
