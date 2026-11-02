"""
render_reflectivity.py

Render a smoothed 3D structure from scattered points with reflectivity values.

Inputs
------
points : (N, 3) array of xyz coordinates
refl   : (N,)   array of reflectivity values in [0, 1]

Pipeline
--------
1. Splat the points into a regular 3D voxel grid (sum of reflectivity + hit count).
2. Smooth both with a Gaussian kernel and divide -> locally averaged reflectivity
   (normalized convolution). A confidence term fades out voxels far from any data.
3. Volume-render with an opacity transfer function:
      v <  low          -> fully transparent
      low <= v <= high  -> opacity ramps up (shaped by `gamma`)
      v >  high         -> max opacity
   so weak reflections vanish and strong ones stand out.

Usage
-----
As a library:
    from render_reflectivity import render_reflectivity
    render_reflectivity(points, refl, low=0.3, high=0.8)

From the command line (with .npy files):
    python render_reflectivity.py --points pts.npy --refl refl.npy --low 0.3 --high 0.8

Demo with synthetic data:
    python render_reflectivity.py --demo

Dependencies:
    pip install numpy scipy pyvista          # default backend (desktop window)
    pip install plotly                       # optional, --backend plotly (browser/HTML)
"""

import argparse
import numpy as np
from scipy.ndimage import gaussian_filter
import h5py
import scipy.io
from path_variables import PATH_TO_0821_GALVO, PATH_TO_LNOI_EDBR_GALVO

# --------------------------------------------------------------------------- #
# 1. Points -> smoothed voxel field
# --------------------------------------------------------------------------- #
def voxelize(points, refl, resolution=128, sigma=1.5, padding=3, mode="mean"):
    """
    Build a smoothed reflectivity volume from scattered points.

    resolution : number of voxels along the longest axis of the bounding box
    sigma      : Gaussian smoothing width, in voxels (bigger = smoother/blobbier)
    padding    : extra voxels around the bounding box so the blur isn't clipped
    mode       : "mean" -> locally averaged reflectivity (robust to point density)
                 "max"  -> local maximum before smoothing (keeps isolated strong
                           returns from being diluted by nearby weak ones)

    Returns (volume, origin, spacing) where volume values are in [0, 1].
    """
    points = np.asarray(points, dtype=float)
    refl = np.clip(np.asarray(refl, dtype=float).ravel(), 0.0, 1.0)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if len(points) != len(refl):
        raise ValueError("points and refl must have the same length")

    pmin, pmax = points.min(axis=0), points.max(axis=0)
    extent = np.maximum(pmax - pmin, 1e-9)
    spacing = extent.max() / (resolution - 1)
    origin = pmin - padding * spacing
    dims = np.ceil(extent / spacing).astype(int) + 1 + 2 * padding

    idx = np.round((points - origin) / spacing).astype(int)
    idx = np.clip(idx, 0, dims - 1)

    if mode == "max":
        vol = np.zeros(dims)
        np.maximum.at(vol, tuple(idx.T), refl)
        smoothed = gaussian_filter(vol, sigma)
        # rescale so a lone point of value r keeps a peak of ~r
        peak = gaussian_filter(_delta(sigma), sigma).max()
        volume = np.clip(smoothed / peak, 0, 1)
        return volume, origin, spacing

    # normalized convolution: blur(sum r) / blur(count)
    num = np.zeros(dims)
    den = np.zeros(dims)
    np.add.at(num, tuple(idx.T), refl)
    np.add.at(den, tuple(idx.T), 1.0)
    num = gaussian_filter(num, sigma)
    den = gaussian_filter(den, sigma)

    mean = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-12)

    # Confidence: ~1 near data, -> 0 far from data. Without this, a voxel
    # 4 sigma away from a single strong point would still read as "strong".
    single_point_peak = 1.0 / ((2 * np.pi) ** 1.5 * sigma ** 3)
    confidence = 1.0 - np.exp(-den / (0.5 * single_point_peak))

    volume = np.clip(mean * confidence, 0, 1)
    return volume, origin, spacing


def normalize_refl(refl, percentile=99.0, log=False):
    """
    Scale raw intensities to [0, 1] using a percentile instead of the max,
    so a few outliers don't compress everything else toward 0.
    Values above the percentile are clipped to 1.
    log=True compresses very heavy-tailed data before scaling.
    """
    refl = np.asarray(refl, dtype=float).ravel()
    if log:
        refl = np.log1p(refl - refl.min())
    lo = refl.min()
    hi = np.percentile(refl, percentile)
    return np.clip((refl - lo) / max(hi - lo, 1e-12), 0.0, 1.0)


def _delta(sigma):
    n = int(np.ceil(4 * sigma)) * 2 + 1
    d = np.zeros((n, n, n))
    d[n // 2, n // 2, n // 2] = 1.0
    return d


# --------------------------------------------------------------------------- #
# 2. Opacity transfer function
# --------------------------------------------------------------------------- #
def opacity_curve(low=0.3, high=0.8, gamma=1.5, max_opacity=0.9, n=256):
    """Opacity sampled over values 0..1: 0 below `low`, ramps to `max_opacity` at `high`."""
    v = np.linspace(0, 1, n)
    t = np.clip((v - low) / max(high - low, 1e-9), 0, 1)
    return max_opacity * t ** gamma


# --------------------------------------------------------------------------- #
# 3. Rendering
# --------------------------------------------------------------------------- #
def render_reflectivity(points, refl, low=0.3, high=0.8, gamma=1.5,
                        max_opacity=0.9, resolution=128, sigma=1.5,
                        mode="mean", cmap="inferno", backend="pyvista",
                        show_points=False, isosurface=None,
                        screenshot=None, html=None):
    """
    Smooth and render. Key knobs:
      low, high    : reflectivity thresholds for the opacity ramp
      gamma        : >1 suppresses mid values more, <1 makes them more visible
      sigma        : smoothing (voxels)
      isosurface   : e.g. 0.7 -> also draw a solid surface at that level
      show_points  : overlay the raw points with refl >= low
    """
    volume, origin, spacing = voxelize(points, refl, resolution, sigma, mode=mode)
    print(f"Volume grid {volume.shape}, voxel size {spacing:.4g}, "
          f"max value {volume.max():.3f}")
    vmax = volume.max()
    if vmax < low:
        print(f"WARNING: the smoothed volume never exceeds low={low} (max {vmax:.3f}), "
              f"so everything is transparent. Lower --low/--high or normalize "
              f"refl with normalize_refl().")
    else:
        frac = (volume > low).mean()
        print(f"{100 * frac:.2f}% of voxels are above low={low}")

    if backend == "pyvista":
        _render_pyvista(volume, origin, spacing, points, refl, low, high, gamma,
                        max_opacity, cmap, show_points, isosurface, screenshot)
    elif backend == "plotly":
        _render_plotly(volume, origin, spacing, low, high, gamma, max_opacity,
                       cmap, isosurface, html)
    else:
        raise ValueError("backend must be 'pyvista' or 'plotly'")
    return volume


def _nice_step(x):
    """Smallest of 1, 2, 5 x 10^k that is >= x."""
    p = 10.0 ** np.floor(np.log10(x))
    return next(m * p for m in (1, 2, 5, 10) if m * p >= x)


def _render_pyvista(volume, origin, spacing, points, refl, low, high, gamma,
                    max_opacity, cmap, show_points, isosurface, screenshot):
    import pyvista as pv

    grid = pv.ImageData(dimensions=volume.shape,
                        spacing=(spacing,) * 3, origin=tuple(origin))
    grid.point_data["reflectivity"] = volume.ravel(order="F")

    pl = pv.Plotter(off_screen=screenshot is not None)
    pl.set_background("white")
    pl.add_volume(grid, scalars="reflectivity", cmap=cmap, clim=[0, 1],
                  opacity=opacity_curve(low, high, gamma, max_opacity),
                  shade=False, scalar_bar_args={"title": "Reflectivity"})

    if isosurface is not None:
        surf = grid.contour([isosurface], scalars="reflectivity")
        if surf.n_points:
            surf = surf.smooth_taubin(n_iter=30)
            pl.add_mesh(surf, color="white", opacity=0.35, smooth_shading=True)

    if show_points:
        keep = np.asarray(refl) >= low
        if keep.any():
            cloud = pv.PolyData(np.asarray(points)[keep])
            cloud["r"] = np.asarray(refl)[keep]
            pl.add_mesh(cloud, scalars="r", cmap=cmap, clim=[0, 1],
                        point_size=3, render_points_as_spheres=True,
                        show_scalar_bar=False)

    pl.add_axes()
    # z-only grid: lines of constant z on the back faces, labelled with z values.
    # The z range is snapped to a round step so the labels are round numbers.
    b = np.array(pl.bounds)
    step = _nice_step((b[5] - b[4]) / 8)
    b[4], b[5] = np.floor(b[4] / step) * step, np.ceil(b[5] / step) * step
    pl.show_grid(bounds=b, show_xaxis=False, show_yaxis=False, ztitle="z (m)",
                 n_zlabels=int(round((b[5] - b[4]) / step)) + 1)
    if screenshot:
        pl.show(screenshot=screenshot)
        print(f"Saved {screenshot}")
    else:
        pl.show()


def _render_plotly(volume, origin, spacing, low, high, gamma, max_opacity,
                   cmap, isosurface, html, max_dim=70):
    import plotly.graph_objects as go
    from scipy.ndimage import zoom

    # Plotly volume rendering is slow for large grids -> downsample
    f = min(1.0, max_dim / max(volume.shape))
    if f < 1.0:
        volume = np.clip(zoom(volume, f, order=1), 0, 1)
        spacing = spacing / f

    nx, ny, nz = volume.shape
    x, y, z = np.meshgrid(origin[0] + spacing * np.arange(nx),
                          origin[1] + spacing * np.arange(ny),
                          origin[2] + spacing * np.arange(nz), indexing="ij")

    # opacityscale is defined over [isomin, isomax] = [0, 1] normalized
    vals = np.linspace(0, 1, 12)
    ops = opacity_curve(low, high, gamma, 1.0, n=12)
    opacityscale = [[float(v), float(o)] for v, o in zip(vals, ops)]

    fig = go.Figure(go.Volume(
        x=x.ravel(), y=y.ravel(), z=z.ravel(), value=volume.ravel(),
        isomin=0.0, isomax=1.0, opacity=max_opacity * 0.3,  # plotly stacks many layers
        opacityscale=opacityscale, surface_count=25,
        colorscale=cmap.capitalize() if cmap == "inferno" else cmap,
        colorbar=dict(title="Reflectivity")))

    if isosurface is not None:
        fig.add_trace(go.Isosurface(
            x=x.ravel(), y=y.ravel(), z=z.ravel(), value=volume.ravel(),
            isomin=isosurface, isomax=isosurface, surface_count=1,
            opacity=0.3, colorscale=[[0, "white"], [1, "white"]],
            showscale=False, caps=dict(x_show=False, y_show=False, z_show=False)))

    fig.update_layout(template="plotly_dark", scene=dict(aspectmode="data"))
    if html:
        fig.write_html(html)
        print(f"Saved {html}")
    else:
        fig.show()


# --------------------------------------------------------------------------- #
# Demo data
# --------------------------------------------------------------------------- #
def make_demo(n=60000, seed=0):
    """Weak noisy background + a strongly reflecting sphere shell and a bright rod."""
    rng = np.random.default_rng(seed)
    bg = rng.uniform(-1.5, 1.5, (n, 3))
    bg_r = rng.uniform(0.0, 0.25, n)

    m = n // 3
    d = rng.normal(size=(m, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    shell = d * (1.0 + rng.normal(0, 0.03, (m, 1)))
    shell_r = np.clip(0.55 + 0.4 * (shell[:, 2] + 1) / 2 + rng.normal(0, 0.05, m), 0, 1)

    k = n // 10
    t = rng.uniform(-1.3, 1.3, k)
    rod = np.c_[0.3 + rng.normal(0, 0.03, k), rng.normal(0, 0.03, k), t]
    rod_r = np.clip(rng.normal(0.95, 0.03, k), 0, 1)

    return np.vstack([bg, shell, rod]), np.concatenate([bg_r, shell_r, rod_r])


# --------------------------------------------------------------------------- #
# >>> YOUR DATA LOADING GOES HERE <<<
# --------------------------------------------------------------------------- #
def load_data(args):
    """
    Load your data and return (points, refl):
        points : (N, 3) array of xyz coordinates
        refl   : (N,)   array of reflectivity values in [0, 1]

    Examples:
        data = np.loadtxt("scan.csv", delimiter=",", skiprows=1)
        points, refl = data[:, :3], data[:, 3]

        points = np.load("points.npy"); refl = np.load("refl.npy")
    """
    # ---- write your loading code below ----
    data = np.load(args.path)

    src = str(data["meta_src"])
    if src.startswith("0821"):
        f = h5py.File(PATH_TO_0821_GALVO, 'r')
        galvo1 = np.array(f["/channels/channel_4"])
        galvo2 = np.array(f["/channels/channel_3"])
        galvo1 -= np.mean(galvo1)
        galvo2 -= np.mean(galvo2) 
        del_t_galvo = 7.328000000000001e-07
        t_galvo = 0.0060001344 + np.arange(131008)*del_t_galvo
        T = 20e-6
        galvo_offset = args.galvo_offset if args.galvo_offset is not None else 3.5e-3 # -19.5e-3?
        galvo_smooth = args.galvo_smooth if args.galvo_smooth is not None else 1e-3
    elif src.startswith("lnoi_edbr"):
        galvo_data = scipy.io.loadmat(PATH_TO_LNOI_EDBR_GALVO)
        galvo1 = galvo_data["Galvo1"][:,0]
        galvo2 = galvo_data["Galvo2"][:,0]
        galvo1 -= np.mean(galvo1)
        galvo2 -= np.mean(galvo2) 
        t_galvo = galvo_data["t_galvo"][:,0]
        del_t_galvo = t_galvo[1] - t_galvo[0]
        T = 10e-6
        galvo_offset = args.galvo_offset if args.galvo_offset is not None else 5e-4
        galvo_smooth = args.galvo_smooth if args.galvo_smooth is not None else 1e-4
    else:
        raise RuntimeError(f"did not recognize {src}")

    segment_t, dist, ps = data["segment_t"], data["distances"], data["peak_strength"]
    if np.ndim(segment_t) == 2:
        segment_t = segment_t.flatten()
        N1, N2, N3 = dist.shape
        dist = np.reshape(dist, (N1*N2, N3))
        ps = np.reshape(ps, (N1*N2, N3))
    else:
        assert(np.ndim(segment_t)==1)

    smoother_len = (int(galvo_smooth / del_t_galvo)//2) * 2 + 1
    if smoother_len > 1:
        galvo1 = np.convolve(galvo1, np.ones((smoother_len,))/smoother_len, mode="same")
        galvo2 = np.convolve(galvo2, np.ones((smoother_len,))/smoother_len, mode="same")
    phi = np.zeros_like(segment_t)
    theta = np.zeros_like(segment_t)
    Galvo_conv = 2/360*2*np.pi
    for i in range(len(segment_t)):
        idx = np.argmin(np.abs(t_galvo-segment_t[i]-galvo_offset))
        if ( np.abs(t_galvo[idx]-segment_t[i]-galvo_offset) > T ):
            dist[i] = np.ones_like(dist[i])*np.nan
        else:
            phi[i] = galvo1[idx]*Galvo_conv
            theta[i] = galvo2[idx]*Galvo_conv

    if args.subtract:
        r = dist.T - args.subtract
    else:
        r = dist.T
    x = r*np.sin(phi)
    y = r*np.cos(phi)*np.sin(theta)
    z = r*np.cos(phi)*np.cos(theta)
    valid_idx = (z>1e-3)*(~np.isnan(r))
    if args.max_z is not None:
        valid_idx *= (z < args.max_z)
    if args.min_z is not None:
        valid_idx *= (z > args.min_z)
    points = np.column_stack([x[valid_idx], y[valid_idx], z[valid_idx]])
    refl = ps.T
    refl = refl[valid_idx]
    refl = normalize_refl(refl, percentile=99)

    # ---- end of your loading code ----
    if points is None or refl is None:
        raise NotImplementedError("Fill in load_data() in render_reflectivity.py")
    return np.asarray(points, dtype=float), np.asarray(refl, dtype=float).ravel()


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--path", default=None, type=str, help="path to result from run.py")
    ap.add_argument("--galvo-offset", default=None, type=float, help="galvo time offset")
    ap.add_argument("--galvo-smooth", default=None, type=float, help="galvo time offset")
    ap.add_argument("--min-z", default=None, type=float)
    ap.add_argument("--max-z", default=None, type=float)
    ap.add_argument("--subtract", default=None, type=float, help="subtract distance")
    ap.add_argument("--demo", action="store_true", help="use synthetic data")
    ap.add_argument("--low", type=float, default=0.3, help="fully transparent below this")
    ap.add_argument("--high", type=float, default=0.8, help="max opacity above this")
    ap.add_argument("--gamma", type=float, default=1.5, help="opacity ramp shape")
    ap.add_argument("--max-opacity", type=float, default=0.9)
    ap.add_argument("--resolution", type=int, default=128)
    ap.add_argument("--sigma", type=float, default=1.5, help="smoothing in voxels")
    ap.add_argument("--mode", choices=["mean", "max"], default="mean")
    ap.add_argument("--cmap", default="inferno")
    ap.add_argument("--backend", choices=["pyvista", "plotly"], default="pyvista")
    ap.add_argument("--isosurface", type=float, default=None)
    ap.add_argument("--show-points", action="store_true")
    ap.add_argument("--screenshot", help="save PNG instead of opening a window (pyvista)")
    ap.add_argument("--html", help="save HTML instead of opening a browser (plotly)")
    a = ap.parse_args()

    if a.demo:
        pts, r = make_demo()
    else:
        pts, r = load_data(a)

    render_reflectivity(pts, r, low=a.low, high=a.high, gamma=a.gamma,
                        max_opacity=a.max_opacity, resolution=a.resolution,
                        sigma=a.sigma, mode=a.mode, cmap=a.cmap, backend=a.backend,
                        show_points=True, isosurface=a.isosurface,
                        screenshot=a.screenshot, html=a.html)
