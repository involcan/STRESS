#!/usr/bin/env python3
"""
brtm.py - Bayesian Right Trihedra Method (BRTM) for stress inversion of focal
mechanisms, after D'Auria & Massa (2015), Seismol. Res. Lett. 86(3),
doi:10.1785/0220140153.

Input : CSV with columns strike,dip,rake (degrees, Aki & Richards convention).
        A header line is optional; extra columns are ignored.
Output: <prefix>_sphere.png   3D focal-sphere view (as Fig. 1c)
        <prefix>_lambert.png  lower-hemisphere equal-area view (as Fig. 1a)
        <prefix>_stress.txt   stress parameters (readable)
        <prefix>_stress.csv   the same, one row (to aggregate many cells)
        <prefix>_grid.csv     marginal PDFs on the sphere grid (--save-grid)

Usage:
    python brtm.py mechs.csv
    python brtm.py mechs.csv -o out/cell01 --lisle off --ngrid 6000

Method (paper equations):
  p(s1,s3) = p1(s1) p3(s3) pL(s1,s3)                                    (3)
  p1, p3   : fraction of mechanisms whose P (resp. T) dihedra contain the
             axis (4,5). With the Aki-Richards moment tensor M = n u^T + u n^T,
             x.M.x < 0 in the P dihedra and > 0 in the T dihedra.
  pL       : fraction of mechanisms satisfying Lisle's (1987) constraint (6).
  p_s1(v)  = integral over psi of p(v, psi)  (marginal, eq. 7); same for s3,
             with the integral done over a Saff & Kuijlaars (1997) grid.
  s1++, s3++: orthogonal pairs maximising p_s1(s1) p_s3(s3);
  s1+, s3+  : average of the (possibly multiple) maxima.
  Phi_B    : grid search over Phi with the axes fixed, choosing for each
             mechanism the nodal plane that best fits the Wallace-Bott
             prediction (in the spirit of Angelier, 1990).
  68% regions: highest-density region of each marginal holding 68% of the
             probability (Jackson & Matsu'ura, 1985).

Lisle's constraint and the nodal plane
---------------------------------------
Wallace-Bott requires b.S.n = 0 on the fault (n normal, b = n x u null axis),
which for sigma1 > sigma2 > sigma3 becomes
    (b.e1)(n.e1) and (b.e3)(n.e3) have the same sign.
This is the plane-free form of eq. (6) (sign of (P x T).v for v1 and v3), but
it holds for the fault plane only: on the auxiliary plane the sign is reversed.
--lisle selects the plane:
    first  : the plane given in the CSV is the fault plane (default)
    second : the auxiliary plane is the fault plane
    off    : pL = 1 (right-dihedra + orthogonality only, no plane needed)
Use "off" when the listed plane is arbitrary (e.g. raw SKHASH/HASH output).
"""
import argparse
import csv
import os
import sys

import numpy as np

# --------------------------------------------------------------------------
# Geometry (NED: x = North, y = East, z = Down)
# --------------------------------------------------------------------------


def sdr2nu(strike, dip, rake):
    """Fault normal n and slip vector u (Aki & Richards, Box 4.4), NED."""
    s, d, r = (np.radians(np.asarray(a, dtype=float)) for a in (strike, dip, rake))
    n = np.stack([-np.sin(d) * np.sin(s), np.sin(d) * np.cos(s), -np.cos(d)], -1)
    u = np.stack([np.cos(r) * np.cos(s) + np.sin(r) * np.cos(d) * np.sin(s),
                  np.cos(r) * np.sin(s) - np.sin(r) * np.cos(d) * np.cos(s),
                  -np.sin(r) * np.sin(d)], -1)
    return n, u


def nu2sdr(n, u):
    """Inverse of sdr2nu (n is flipped upward if needed)."""
    n = np.atleast_2d(np.asarray(n, dtype=float)).copy()
    u = np.atleast_2d(np.asarray(u, dtype=float)).copy()
    flip = n[:, 2] > 0
    n[flip] *= -1
    u[flip] *= -1
    dip = np.degrees(np.arccos(np.clip(-n[:, 2], -1, 1)))
    strike = np.degrees(np.arctan2(-n[:, 0], n[:, 1])) % 360
    sr = np.radians(strike)
    sv = np.stack([np.cos(sr), np.sin(sr), np.zeros_like(sr)], -1)
    sind = np.maximum(np.sin(np.radians(dip)), 1e-12)
    rake = np.degrees(np.arctan2(-u[:, 2] / sind, (u * sv).sum(1)))
    return strike, dip, rake


def vec2tp(v):
    """Trend/plunge (deg) of an axis, lower hemisphere."""
    v = np.asarray(v, dtype=float)
    v = v / np.linalg.norm(v)
    if v[2] < 0:
        v = -v
    return np.degrees(np.arctan2(v[1], v[0])) % 360, np.degrees(np.arcsin(np.clip(v[2], -1, 1)))


def tp2vec(trend, plunge):
    t, p = np.radians(trend), np.radians(plunge)
    return np.array([np.cos(p) * np.cos(t), np.cos(p) * np.sin(t), np.sin(p)])


def sphere_points(n):
    """Saff & Kuijlaars (1997) spiral: n nearly uniform points on the sphere."""
    k = np.arange(n)
    h = -1 + 2 * k / (n - 1)
    theta = np.arccos(h)
    inc = np.zeros(n)
    inc[1:-1] = 3.6 / np.sqrt(n) / np.sqrt(1 - h[1:-1] ** 2)
    phi = np.cumsum(inc) % (2 * np.pi)
    phi[-1] = 0.0
    return np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), h], -1)


def perp_basis(v):
    """Two unit vectors orthogonal to each row of v (and to each other)."""
    a = np.zeros_like(v)
    a[:, 2] = 1.0
    near = np.abs(v[:, 2]) > 0.9
    a[near] = [1.0, 0.0, 0.0]
    e1 = np.cross(v, a)
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    e2 = np.cross(v, e1)
    return e1, e2


def axial_mean(vecs, weights=None):
    """Mean axis (principal eigenvector of the orientation tensor)."""
    vecs = np.atleast_2d(vecs)
    w = np.ones(len(vecs)) if weights is None else np.asarray(weights)
    T = (vecs * w[:, None]).T @ vecs
    val, vec = np.linalg.eigh(T)
    return vec[:, -1]


class NearestGrid:
    """Nearest-neighbour / IDW lookup of values defined on sphere points."""

    def __init__(self, grid):
        self.grid = grid
        try:
            from scipy.spatial import cKDTree
            self.tree = cKDTree(grid)
        except ImportError:
            self.tree = None

    def query(self, x, k=1):
        if self.tree is not None:
            return self.tree.query(x, k=k)
        # fallback: brute force by chunks
        dist, idx = [], []
        for i in range(0, len(x), 4000):
            d2 = 2 - 2 * (x[i:i + 4000] @ self.grid.T)
            if k == 1:
                j = np.argmin(d2, 1)
                idx.append(j)
                dist.append(np.sqrt(np.maximum(d2[np.arange(len(j)), j], 0)))
            else:
                j = np.argpartition(d2, k, axis=1)[:, :k]
                idx.append(j)
                dist.append(np.sqrt(np.maximum(np.take_along_axis(d2, j, 1), 0)))
        return np.concatenate(dist), np.concatenate(idx)

    def nearest(self, values, x):
        _, i = self.query(x, 1)
        return values[i]

    def idw(self, values, x, k=12):
        d, i = self.query(x, k)
        w = 1.0 / (d + 1e-6) ** 2
        return (values[i] * w).sum(1) / w.sum(1)


# --------------------------------------------------------------------------
# BRTM
# --------------------------------------------------------------------------
class BRTM:
    def __init__(self, strike, dip, rake, lisle="first", ngrid=4000, npsi=72,
                 conf=0.68, rtol=1e-6, chunk_elems=2e7):
        self.sdr = np.column_stack([strike, dip, rake]).astype(float)
        self.n, self.u = sdr2nu(strike, dip, rake)
        self.K = len(self.n)
        self.T = (self.n + self.u) / np.sqrt(2)
        self.P = (self.n - self.u) / np.sqrt(2)
        self.b = np.cross(self.n, self.u)
        if lisle not in ("first", "second", "off"):
            raise ValueError("lisle must be 'first', 'second' or 'off'")
        self.lisle = lisle
        self.f = self.n if lisle == "first" else self.u
        self.G = sphere_points(ngrid)
        self.lookup = NearestGrid(self.G)
        self.npsi = npsi
        self.conf = conf
        self.rtol = rtol
        self.chunk_elems = chunk_elems

    # x.M.x sign: < 0 -> P dihedra (sigma1), > 0 -> T dihedra (sigma3)
    def _dihedra(self, X):
        return (X @ self.n.T) * (X @ self.u.T)

    def _lisle_sign(self, X):
        return np.sign((X @ self.b.T) * (X @ self.f.T))

    def _rotated(self, V):
        """Axes orthogonal to V, sampled on psi in [0, pi): shape (C, npsi, 3)."""
        e1, e2 = perp_basis(V)
        psi = (np.arange(self.npsi) + 0.5) * np.pi / self.npsi
        return (np.cos(psi)[None, :, None] * e1[:, None, :]
                + np.sin(psi)[None, :, None] * e2[:, None, :])

    def _marginal(self, which):
        """Marginal p_sigma(which) on the grid, integrating over psi (eq. 7)."""
        G, K, npsi = self.G, self.K, self.npsi
        dg = self._dihedra(G)
        pg = (dg < 0).mean(1) if which == 1 else (dg > 0).mean(1)
        lg = self._lisle_sign(G) if self.lisle != "off" else None
        out = np.zeros(len(G))
        step = max(1, int(self.chunk_elems // (npsi * K)))
        for i0 in range(0, len(G), step):
            idx = np.arange(i0, min(i0 + step, len(G)))
            W = self._rotated(G[idx])
            Wf = W.reshape(-1, 3)
            dw = self._dihedra(Wf)
            pw = (dw > 0).mean(1) if which == 1 else (dw < 0).mean(1)
            pw = pw.reshape(len(idx), npsi)
            if lg is not None:
                lw = self._lisle_sign(Wf).reshape(len(idx), npsi, K)
                pl = (lw * lg[idx][:, None, :] > 0).mean(2)
            else:
                pl = 1.0
            out[idx] = (pg[idx][:, None] * pw * pl).mean(1)
        return out

    def _hdr_threshold(self, p):
        """Density level enclosing a fraction conf of the probability."""
        s = np.sort(p)[::-1]
        c = np.cumsum(s) / s.sum()
        return s[min(np.searchsorted(c, self.conf), len(s) - 1)]

    def run(self):
        self.ps1 = self._marginal(1)
        self.ps3 = self._marginal(3)
        self.thr1 = self._hdr_threshold(self.ps1)
        self.thr3 = self._hdr_threshold(self.ps3)

        # sigma++ : orthogonal pairs maximising p_s1(s1) p_s3(s3)
        best, cand1, cand3 = -1.0, [], []
        step = max(1, int(2e6 // self.npsi))
        for i0 in range(0, len(self.G), step):
            idx = np.arange(i0, min(i0 + step, len(self.G)))
            W = self._rotated(self.G[idx])
            p3w = self.lookup.nearest(self.ps3, W.reshape(-1, 3)).reshape(len(idx), -1)
            p13 = self.ps1[idx][:, None] * p3w
            m = p13.max()
            if m > best * (1 + self.rtol):
                best, cand1, cand3 = m, [], []
            if m >= best * (1 - self.rtol):
                ii, jj = np.nonzero(p13 >= best * (1 - self.rtol))
                cand1.append(self.G[idx][ii])
                cand3.append(W[ii, jj])
        cand1, cand3 = np.vstack(cand1), np.vstack(cand3)
        s1 = axial_mean(cand1)
        s3 = axial_mean(cand3)
        s3 = s3 - (s3 @ s1) * s1
        s3 /= np.linalg.norm(s3)
        s2 = np.cross(s3, s1)
        self.s1, self.s2, self.s3 = s1, s2, s3
        self.n_maxima = len(cand1)
        self.phi, self.phi_misfit, self.plane_choice = self.bishop_ratio()
        return self

    def bishop_ratio(self, nphi=201):
        """Phi_B = (s2-s3)/(s1-s3) with fixed axes; best nodal plane per event."""
        e1, e2 = self.s1, self.s2
        phis = np.linspace(0, 1, nphi)
        mis = np.zeros((nphi, self.K))
        choice = np.zeros((nphi, self.K), dtype=int)
        for k, phi in enumerate(phis):
            S = np.outer(e1, e1) + phi * np.outer(e2, e2)   # compression positive, s3 = 0
            angs = []
            for f, s in ((self.n, self.u), (self.u, self.n)):
                t = -(f @ S)
                sh = t - (t * f).sum(1, keepdims=True) * f
                nrm = np.linalg.norm(sh, axis=1)
                c = np.where(nrm > 1e-12, (sh * s).sum(1) / np.maximum(nrm, 1e-12), -1)
                angs.append(np.degrees(np.arccos(np.clip(c, -1, 1))))
            angs = np.array(angs)
            choice[k] = angs.argmin(0)
            mis[k] = angs.min(0)
        j = mis.mean(1).argmin()
        return phis[j], mis[j].mean(), choice[j]

    def conf_radius(self, which):
        """Angular radius (deg) of a cone with the same area as the 68% region."""
        p, thr = (self.ps1, self.thr1) if which == 1 else (self.ps3, self.thr3)
        frac = (p >= thr).mean()           # fraction of the sphere (both antipodes)
        return np.degrees(np.arccos(np.clip(1 - frac, -1, 1)))


# --------------------------------------------------------------------------
# Colours (Fig. 1b): bilinear palette of p^_s1 (x) and p^_s3 (y)
# --------------------------------------------------------------------------
C00 = np.array([0.98, 0.97, 0.98])   # low both (sigma2-like)
C10 = np.array([0.90, 0.12, 0.30])   # sigma1
C01 = np.array([0.22, 0.36, 0.86])   # sigma3
C11 = np.array([0.80, 0.22, 0.78])   # both (undetermined)


def palette(a, b):
    a = np.clip(a, 0, 1)[..., None]
    b = np.clip(b, 0, 1)[..., None]
    return ((1 - a) * (1 - b) * C00 + a * (1 - b) * C10
            + (1 - a) * b * C01 + a * b * C11)


def add_palette_inset(fig, rect):
    ax = fig.add_axes(rect)
    g = np.linspace(0, 1, 64)
    A, B = np.meshgrid(g, g)
    ax.imshow(palette(A, B), origin="lower", extent=(0, 1, 0, 1))
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xlabel(r"$\hat p_{\sigma_1}$", fontsize=8, labelpad=-6)
    ax.set_ylabel(r"$\hat p_{\sigma_3}$", fontsize=8, labelpad=-6)
    ax.tick_params(labelsize=7, length=2)
    return ax


def summary_text(r):
    t1, p1 = vec2tp(r.s1)
    t3, p3 = vec2tp(r.s3)
    return (r"$\sigma_1^\dagger$ {:05.1f}/{:04.1f}   "
            r"$\sigma_3^\dagger$ {:05.1f}/{:04.1f}   "
            r"$\Phi_B$={:.2f}   N={}").format(t1, p1, t3, p3, r.phi, r.K)


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
def _contour_lines(x, y, z, level):
    try:
        import contourpy
        return [l for l in contourpy.contour_generator(x, y, z).lines(level)]
    except Exception:
        import matplotlib.pyplot as plt
        f = plt.figure()
        cs = plt.contour(x, y, z, [level])
        segs = [p.vertices for p in cs.get_paths()] if hasattr(cs, "get_paths") else cs.allsegs[0]
        plt.close(f)
        return segs


def plot_sphere(r, path, elev=25, azim=-55, title=None, dpi=200):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m1, m3 = r.ps1.max(), r.ps3.max()
    nth, nph = 121, 241
    th = np.linspace(0, np.pi, nth)            # colatitude from Up
    ph = np.linspace(0, 2 * np.pi, nph)        # from East, counter-clockwise
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    E = np.sin(TH) * np.cos(PH)
    N = np.sin(TH) * np.sin(PH)
    U = np.cos(TH)
    ned = np.stack([N, E, -U], -1).reshape(-1, 3)
    v1 = r.lookup.idw(r.ps1, ned).reshape(TH.shape)
    v3 = r.lookup.idw(r.ps3, ned).reshape(TH.shape)
    cols = palette(v1 / m1, v3 / m3)

    fig = plt.figure(figsize=(6.4, 6.4))
    ax = fig.add_subplot(projection="3d", computed_zorder=False)
    ax.plot_surface(E, N, U, facecolors=cols, rstride=1, cstride=1,
                    shade=False, linewidth=0, antialiased=False, zorder=1)

    ev, az = np.radians(elev), np.radians(azim)
    view = np.array([np.cos(ev) * np.cos(az), np.cos(ev) * np.sin(az), np.sin(ev)])

    def visible(P):
        """Mask (NaN) points hidden by the unit sphere."""
        P = np.array(P, dtype=float)
        dv = P @ view
        rproj = np.sqrt(np.maximum((P ** 2).sum(-1) - dv ** 2, 0))
        hidden = (dv < 0) & (rproj < 1.0)
        P[hidden] = np.nan
        return P

    # 68% contours
    for vals, thr, col in ((v1, r.thr1, C10), (v3, r.thr3, C01)):
        for seg in _contour_lines(PH, TH, vals, thr):
            if len(seg) < 2:
                continue
            pp, tt = seg[:, 0], seg[:, 1]
            P = 1.004 * np.stack([np.sin(tt) * np.cos(pp), np.sin(tt) * np.sin(pp), np.cos(tt)], -1)
            P[(P @ view) < 0] = np.nan
            ax.plot(P[:, 0], P[:, 1], P[:, 2], color=col * 0.75, lw=1.4, zorder=4)

    def enu(v):
        return np.array([v[1], v[0], -v[2]])

    def stick(v, color, label, length=1.45):
        d = enu(v)
        for sgn in (1, -1):
            t = np.linspace(1.0, length, 30)[:, None]
            P = visible(sgn * t * d)
            ax.plot(P[:, 0], P[:, 1], P[:, 2], color=color, lw=3, zorder=6,
                    solid_capstyle="round")
        tip = d if d @ view >= 0 else -d
        ax.text(*(tip * (length + 0.12)), label, color=color, fontsize=12,
                ha="center", va="center", zorder=7)

    stick(r.s1, C10 * 0.85, r"$\sigma_1^\dagger$")
    stick(r.s3, C01 * 0.85, r"$\sigma_3^\dagger$")

    for v, lab in (((0, 1, 0), "N"), ((1, 0, 0), "E"), ((0, 0, 1), "V")):
        v = np.array(v, float)
        P = visible(np.linspace(1.0, 1.25, 20)[:, None] * v)
        ax.plot(P[:, 0], P[:, 1], P[:, 2], color="k", lw=1.5, zorder=5)
        ax.text(*(v * 1.38), lab, fontsize=12, ha="center", va="center", zorder=7)

    ax.view_init(elev=elev, azim=azim)
    lim = 1.3
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1, 1, 1), zoom=1.35)
    ax.set_axis_off()
    ax.set_position([0, 0.06, 1, 0.9])
    fig.text(0.5, 0.04, summary_text(r), ha="center", fontsize=10)
    if title:
        fig.text(0.5, 0.97, title, ha="center", fontsize=12)
    add_palette_inset(fig, [0.83, 0.83, 0.12, 0.12])
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def lambert_xy(v):
    """Lower-hemisphere equal-area projection (unit radius), x = East, y = North."""
    v = np.atleast_2d(v).astype(float).copy()
    v[v[:, 2] < 0] *= -1
    theta = np.arccos(np.clip(v[:, 2], -1, 1))
    R = np.sqrt(2) * np.sin(theta / 2)
    az = np.arctan2(v[:, 1], v[:, 0])
    return R * np.sin(az), R * np.cos(az)


def plot_lambert(r, path, title=None, dpi=200, npix=500):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = np.linspace(-1, 1, npix)
    X, Y = np.meshgrid(g, g)
    R = np.hypot(X, Y)
    inside = R <= 1
    theta = 2 * np.arcsin(np.clip(R, 0, 1) / np.sqrt(2))
    az = np.arctan2(X, Y)
    V = np.stack([np.sin(theta) * np.cos(az), np.sin(theta) * np.sin(az), np.cos(theta)], -1)
    v1 = r.lookup.idw(r.ps1, V.reshape(-1, 3)).reshape(X.shape)
    v3 = r.lookup.idw(r.ps3, V.reshape(-1, 3)).reshape(X.shape)
    rgba = np.ones(X.shape + (4,))
    rgba[..., :3] = palette(v1 / r.ps1.max(), v3 / r.ps3.max())
    rgba[~inside, 3] = 0

    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    ax.imshow(rgba, origin="lower", extent=(-1, 1, -1, 1), interpolation="bilinear")
    v1m = np.where(inside, v1, np.nan)
    v3m = np.where(inside, v3, np.nan)
    ax.contour(X, Y, v1m, [r.thr1], colors=[C10 * 0.75], linewidths=1.5)
    ax.contour(X, Y, v3m, [r.thr3], colors=[C01 * 0.75], linewidths=1.5)
    t = np.linspace(0, 2 * np.pi, 361)
    ax.plot(np.cos(t), np.sin(t), "k", lw=1)
    ax.plot([0, 0], [1, 1.05], "k", lw=1)
    ax.text(0, 1.09, "N", ha="center", va="bottom", fontsize=12)
    for v, col in ((r.s1, C10), (r.s3, C01)):
        x, y = lambert_xy(v)
        ax.plot(x, y, "^", ms=13, mfc=col, mec="k", mew=1.2)
    ax.set_xlim(-1.12, 1.12)
    ax.set_ylim(-1.12, 1.18)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(-1.1, 1.1, r"$\Phi_B$={:.2f}".format(r.phi), fontsize=11)
    fig.text(0.5, 0.04, summary_text(r), ha="center", fontsize=9)
    if title:
        ax.set_title(title)
    add_palette_inset(fig, [0.80, 0.80, 0.11, 0.11])
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------
def read_sdr(path):
    rows, cols = [], None
    with open(path, newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t ")
        except csv.Error:
            dialect = csv.excel
        for row in csv.reader(f, dialect):
            row = [c.strip() for c in row if c.strip() != ""]
            if not row or row[0].startswith("#"):
                continue
            try:
                vals = [float(c) for c in row]
            except ValueError:
                if cols is None:                       # header
                    low = [c.lower() for c in row]
                    if {"strike", "dip", "rake"}.issubset(low):
                        cols = [low.index(c) for c in ("strike", "dip", "rake")]
                continue
            ix = cols if cols is not None else [0, 1, 2]
            rows.append([vals[i] for i in ix])
    a = np.array(rows, dtype=float)
    if a.ndim != 2 or a.shape[0] == 0:
        sys.exit("No strike,dip,rake rows found in {}".format(path))
    return a[:, 0], a[:, 1], a[:, 2]


def write_results(r, prefix, infile):
    t1, p1 = vec2tp(r.s1)
    t2, p2 = vec2tp(r.s2)
    t3, p3 = vec2tp(r.s3)
    c1, c3 = r.conf_radius(1), r.conf_radius(3)
    first = (r.plane_choice == 0).mean()
    with open(prefix + "_stress.txt", "w") as f:
        f.write("# BRTM stress inversion (D'Auria & Massa, 2015)\n")
        f.write("input              : {}\n".format(infile))
        f.write("n_events           : {}\n".format(r.K))
        f.write("lisle_constraint   : {}\n".format(r.lisle))
        f.write("grid_points / psi  : {} / {}\n".format(len(r.G), r.npsi))
        f.write("\n# axis       trend   plunge\n")
        f.write("sigma1       {:6.1f}   {:5.1f}\n".format(t1, p1))
        f.write("sigma2       {:6.1f}   {:5.1f}\n".format(t2, p2))
        f.write("sigma3       {:6.1f}   {:5.1f}\n".format(t3, p3))
        f.write("\nphi_B              : {:.3f}\n".format(r.phi))
        f.write("mean_misfit_deg    : {:.1f}   (Wallace-Bott, best nodal plane)\n".format(r.phi_misfit))
        f.write("first_plane_chosen : {:.0f}%\n".format(100 * first))
        f.write("conf{:02.0f}_radius_s1 : {:.1f} deg (equal-area cone)\n".format(100 * r.conf, c1))
        f.write("conf{:02.0f}_radius_s3 : {:.1f} deg (equal-area cone)\n".format(100 * r.conf, c3))
        f.write("max_p_sigma1       : {:.4g}\n".format(r.ps1.max()))
        f.write("max_p_sigma3       : {:.4g}\n".format(r.ps3.max()))
        f.write("n_maxima_averaged  : {}\n".format(r.n_maxima))
    with open(prefix + "_stress.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["input", "n_events", "s1_trend", "s1_plunge", "s2_trend", "s2_plunge",
                    "s3_trend", "s3_plunge", "phi_B", "misfit_deg",
                    "conf_radius_s1", "conf_radius_s3", "lisle"])
        w.writerow([infile, r.K, "%.1f" % t1, "%.1f" % p1, "%.1f" % t2, "%.1f" % p2,
                    "%.1f" % t3, "%.1f" % p3, "%.3f" % r.phi, "%.1f" % r.phi_misfit,
                    "%.1f" % c1, "%.1f" % c3, r.lisle])


def write_grid(r, prefix):
    with open(prefix + "_grid.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x_north", "y_east", "z_down", "trend", "plunge", "p_sigma1", "p_sigma3"])
        for v, a, b in zip(r.G, r.ps1, r.ps3):
            t, p = np.degrees(np.arctan2(v[1], v[0])) % 360, np.degrees(np.arcsin(v[2]))
            w.writerow(["%.5f" % v[0], "%.5f" % v[1], "%.5f" % v[2],
                        "%.2f" % t, "%.2f" % p, "%.6g" % a, "%.6g" % b])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="CSV with strike,dip,rake")
    ap.add_argument("-o", "--prefix", help="output prefix (default: input name)")
    ap.add_argument("--lisle", choices=["first", "second", "off"], default="first",
                    help="nodal plane used for Lisle's constraint (default: first)")
    ap.add_argument("--ngrid", type=int, default=4000, help="sphere grid points (default 4000)")
    ap.add_argument("--npsi", type=int, default=72, help="psi samples in [0,180) (default 72)")
    ap.add_argument("--conf", type=float, default=0.68, help="confidence level (default 0.68)")
    ap.add_argument("--elev", type=float, default=25, help="3D view elevation")
    ap.add_argument("--azim", type=float, default=-55, help="3D view azimuth")
    ap.add_argument("--title", help="plot title")
    ap.add_argument("--no-lambert", action="store_true", help="skip the Lambert plot")
    ap.add_argument("--save-grid", action="store_true", help="write the marginal PDFs")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    s, d, rk = read_sdr(args.csv)
    prefix = args.prefix or os.path.splitext(args.csv)[0]
    if os.path.dirname(prefix):
        os.makedirs(os.path.dirname(prefix), exist_ok=True)

    r = BRTM(s, d, rk, lisle=args.lisle, ngrid=args.ngrid, npsi=args.npsi,
             conf=args.conf).run()
    write_results(r, prefix, args.csv)
    if args.save_grid:
        write_grid(r, prefix)
    plot_sphere(r, prefix + "_sphere.png", elev=args.elev, azim=args.azim,
                title=args.title, dpi=args.dpi)
    if not args.no_lambert:
        plot_lambert(r, prefix + "_lambert.png", title=args.title, dpi=args.dpi)

    with open(prefix + "_stress.txt") as f:
        print(f.read())


if __name__ == "__main__":
    main()
