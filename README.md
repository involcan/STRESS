# brtm — Bayesian Right Trihedra Method

Python implementation of the **Bayesian Right Trihedra Method (BRTM)** for
stress inversion of earthquake focal mechanisms:

> D'Auria, L., and B. Massa (2015). Stress inversion of focal mechanism data
> using a Bayesian approach: a novel formulation of the right trihedra method.
> *Seismological Research Letters* 86(3), doi:10.1785/0220140153.

From a set of focal mechanisms (strike, dip, rake), BRTM computes the marginal
probability distributions of the σ1 and σ3 axes over the focal sphere. From
these it derives the best-estimate principal axes σ1†, σ2†, σ3†, the 68%
confidence regions and the Bishop ratio Φ_B. The marginals and confidence
regions are computed without choosing a fault plane; only the Lisle constraint
(`--lisle first|second`) and the Φ_B estimate depend on the nodal plane. The
results are drawn as in Fig. 1 of the paper.

## Requirements

- Python ≥ 3.8
- numpy
- matplotlib (≥ 3.5)
- scipy and contourpy (both optional): used for faster nearest-neighbour
  lookup and for the contours on the sphere. If they are missing, the script
  uses slower fallbacks. contourpy ships with matplotlib ≥ 3.6.

```bash
pip install numpy matplotlib scipy
```

## Quick start

```bash
python brtm.py mechs.csv
python brtm.py mechs.csv -o results/cell01 --title "Cell 01"
python brtm.py mechs.csv --lisle off --ngrid 6000 --save-grid
```

### Input

A CSV file with one focal mechanism per row. Strike, dip and rake are in
degrees and follow the Aki & Richards convention:

```
strike,dip,rake
146.8,76.9,-58.0
107.3,57.7,-118.5
...
```

- The header line is optional. If there is one, the `strike`, `dip` and `rake`
  columns are found by name, and any other columns are ignored.
- Without a header, the script uses the first three columns.
- Commas, semicolons, tabs and spaces are all accepted as delimiters.
- Lines starting with `#` are skipped.

### Output

All files are written as `<prefix>_*`. The prefix defaults to the input file
name without its extension; set it with `-o`.

| File | Content |
|---|---|
| `<prefix>_sphere.png` | 3D view of the focal sphere (as Fig. 1c of the paper) |
| `<prefix>_lambert.png` | Lower-hemisphere Lambert equal-area projection (as Fig. 1a) |
| `<prefix>_stress.txt` | Stress parameters, human-readable |
| `<prefix>_stress.csv` | The same parameters as a single row, for stacking runs on many cells |
| `<prefix>_grid.csv` | Marginal PDFs p_σ1 and p_σ3 on the sphere grid (only with `--save-grid`) |

Example `_stress.txt`:

```
# BRTM stress inversion (D'Auria & Massa, 2015)
input              : normal_NE.csv
n_events           : 52
lisle_constraint   : first
grid_points / psi  : 4000 / 72

# axis       trend   plunge
sigma1        110.0    83.0
sigma2        311.5     6.5
sigma3        221.2     2.6

phi_B              : 0.440
mean_misfit_deg    : 8.4   (Wallace-Bott, best nodal plane)
first_plane_chosen : 90%
conf68_radius_s1 : 43.5 deg (equal-area cone)
conf68_radius_s3 : 49.2 deg (equal-area cone)
max_p_sigma1       : 0.384
max_p_sigma3       : 0.3626
n_maxima_averaged  : 1
```

Fields:

- **Axes.** Trend and plunge in degrees, on the lower hemisphere.
- **`phi_B`.** Bishop ratio, (σ2 − σ3)/(σ1 − σ3).
- **`mean_misfit_deg`.** Mean angle between the observed slip vectors and the
  shear traction predicted by the inverted tensor. For each event the
  better-fitting nodal plane is used.
- **`first_plane_chosen`.** Percentage of events for which the plane listed in
  the CSV fits better than the auxiliary plane.
- **`conf68_radius_*`.** Angular radius of a circular cone with the same area
  as the 68% confidence region. This is a single-number size of the region;
  the region itself need not be circular.
- **`n_maxima_averaged`.** Number of orthogonal pairs (σ1‡, σ3‡) tied at the
  maximum and averaged into σ1†, σ3†.

### Reading the plots

The colour of each point on the sphere combines the normalised marginal
probabilities p̂_σ1 (horizontal axis of the palette inset) and p̂_σ3 (vertical
axis):

- **Red:** probable σ1.
- **Blue:** probable σ3.
- **Magenta:** σ1 and σ3 about equally probable, so the direction is
  undetermined.
- **White:** both probabilities low, which is where σ2 is expected.

The red and blue lines are the boundaries of the 68% confidence regions for
σ1 and σ3. σ1† and σ3† are drawn as red and blue sticks in the 3D view and as
red and blue triangles in the Lambert projection.

## Options

| Option | Default | Description |
|---|---|---|
| `-o, --prefix` | input name | Prefix for the output files (the directory is created if needed) |
| `--lisle {first,second,off}` | `first` | Nodal plane used for Lisle's constraint (see below) |
| `--ngrid` | 4000 | Number of points of the Saff & Kuijlaars grid on the sphere |
| `--npsi` | 72 | Number of samples of the rotation angle ψ over [0°, 180°) |
| `--conf` | 0.68 | Probability enclosed by the confidence regions |
| `--elev`, `--azim` | 25, −55 | Viewpoint of the 3D plot (degrees) |
| `--title` | – | Title added to the plots |
| `--no-lambert` | – | Skip the Lambert projection |
| `--save-grid` | – | Also write the marginal PDFs to `_grid.csv` |
| `--dpi` | 200 | Resolution of the PNG files |

Run time grows with `ngrid × npsi × N_events`. With the default settings,
50–100 mechanisms take a few seconds. For several thousand mechanisms you can
reduce `--ngrid` and `--npsi`.

## Method

The joint probability of a pair of orthogonal axes (σ1, σ3) is (eq. 3 of the
paper)

  p(σ1, σ3) = p1(σ1) · p3(σ3) · pL(σ1, σ3)

with:

- **p1, p3 (eqs. 4–5).** p1 is the fraction of mechanisms whose P dihedra
  contain σ1, and p3 the fraction whose T dihedra contain σ3. The code builds
  the moment tensor from the Aki & Richards fault normal **n** and slip vector
  **u** as M = **n u**ᵀ + **u n**ᵀ. A direction **x** is in the P dihedra when
  **x**·M·**x** < 0 and in the T dihedra when **x**·M·**x** > 0.
- **pL (eq. 6).** The fraction of mechanisms that satisfy Lisle's (1987)
  constraint.

The other steps:

1. **Marginals (eq. 7).** For each direction on a Saff & Kuijlaars (1997)
   grid, the joint probability is integrated over ψ, the rotation of the
   companion axis about it. This gives p_σ1; p_σ3 is computed the same way.
2. **σ‡ and σ†.** σ1‡ and σ3‡ are the orthogonal pairs that maximise
   p_σ1(σ1‡) · p_σ3(σ3‡). When several pairs tie at the maximum, they are
   averaged (axial mean) to give σ1† and σ3†. σ3† is then made exactly
   orthogonal to σ1†, and σ2† = σ3† × σ1†.
3. **Confidence regions.** For each marginal, the region is the set of
   highest-density points that together hold 68% of the probability (Jackson &
   Matsu'ura, 1985).
4. **Bishop ratio.** With the axes fixed, Φ_B is found by grid search. For each
   mechanism the nodal plane that best fits the Wallace–Bott prediction is
   used, and Φ_B minimises the mean misfit angle. This follows the idea of
   Angelier (1990), but it is a grid search, not the analytical solution of
   his Appendix III.

### Lisle's constraint and the choice of nodal plane

The Wallace–Bott hypothesis requires the shear traction on the fault to have
no component along the null axis **b** = **n** × **u**. For σ1 > σ2 > σ3 this
becomes:

  (**b**·**e1**)(**n**·**e1**) and (**b**·**e3**)(**n**·**e3**) have the same
  sign.

This is the form of eq. (6) that does not depend on the arbitrary sign of the
vectors. It holds only when **n** is the normal to the actual fault plane: on
the auxiliary plane the sign is reversed. For this reason the code needs to
know which of the two planes is the fault:

- **`--lisle first`** (default): the plane given in the CSV is the fault plane.
  Use this when the planes have been selected beforehand, for example from
  geology, from relocated seismicity or with a stability criterion.
- **`--lisle second`**: the auxiliary plane is the fault plane.
- **`--lisle off`**: pL = 1. The inversion then uses only the right-dihedra
  condition and the orthogonality of the axes, so no plane choice is needed.
  Use this when the plane listed first is arbitrary, as in raw SKHASH or HASH
  output.

On synthetic data without noise, generated with the Wallace–Bott hypothesis,
the constraint holds for 100% of the events when the fault plane is listed
first, and for about 50% when the order of the planes is random.

## Validation

The script was tested on synthetic data. Slip vectors were generated with the
Wallace–Bott hypothesis on well-oriented planes, and strike, dip and rake were
then perturbed with Gaussian noise (σ = 8°):

| Dataset | True σ1 / σ3 / Φ_B | Recovered σ1 / σ3 / Φ_B |
|---|---|---|
| 52 normal faults, NE–SW extension | 90° plunge / 045°/0° / 0.50 | 83° plunge / 221°/3° / 0.44 |
| 60 strike-slip faults | 030°/10° / 120°/5° / 0.30 | 028°/6° / 118°/4° / 0.35 |
| 104 events, NE–SW + NW–SE extension | – | σ1 vertical; σ3 spread around the horizontal plane |

The heterogeneous case behaves as in Fig. 2l of the paper: σ1 is well
constrained, while the uncertainty on σ3 shows up as a broad blue band around
the horizontal plane.

## Using it from Python

```python
from brtm import BRTM, read_sdr, vec2tp, plot_sphere

strike, dip, rake = read_sdr("mechs.csv")
r = BRTM(strike, dip, rake, lisle="off", ngrid=4000, npsi=72).run()

print(vec2tp(r.s1), vec2tp(r.s3), r.phi)   # axes (trend, plunge) and Phi_B
r.ps1, r.ps3                               # marginal PDFs on the grid r.G (NED)
plot_sphere(r, "cell01_sphere.png", title="Cell 01")
```

Vectors are in the NED frame (x = North, y = East, z = Down).

## References

- Angelier, J. (1990). Inversion of field data in fault tectonics to obtain the
  regional stress — III. A new rapid direct inversion method by analytical
  means. *Geophys. J. Int.* 103, 363–376.
- D'Auria, L., and B. Massa (2015). Stress inversion of focal mechanism data
  using a Bayesian approach: a novel formulation of the right trihedra method.
  *Seismol. Res. Lett.* 86(3), doi:10.1785/0220140153.
- Jackson, D. D., and M. Matsu'ura (1985). A Bayesian approach to nonlinear
  inversion. *J. Geophys. Res.* 90, 581–591.
- Lisle, R. J. (1987). Principal stress orientations from faults: an additional
  constraint. *Annales Tectonicæ* 1, 155–158.
- Saff, E. B., and A. B. J. Kuijlaars (1997). Distributing many points on a
  sphere. *Math. Intelligencer* 19, 5–11.
