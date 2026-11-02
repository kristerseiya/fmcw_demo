#!/usr/bin/env python3
"""
slice_viewer.py - Interactive orthogonal cross-section viewer for 3D arrays.

Shows three orthogonal slices through an (N1, N2, N3) array:
    panel 1: arr[i, :, :]   (axis 0 fixed at i)
    panel 2: arr[:, j, :]   (axis 1 fixed at j)
    panel 3: arr[:, :, k]   (axis 2 fixed at k)
Red crosshairs in each panel mark where the other two slices cut through it.

Navigation
    index sliders      set i, j, k directly
    left click         on a panel to move the other two slices to that point
    scroll wheel       over a panel to step through that panel's slices
    up/down arrows     same as scrolling (page up/down jumps by 10)
    toolbar zoom/pan   works as usual (clicks are ignored while it's active)

Color limits (bottom right)
    Global             one vmin/vmax shared by all three panels. Drag the range
                       slider, or type exact values in the vmin/vmax boxes and
                       press Enter. "Data min/max" resets to the array's full range.
    Per slice          each panel scales to the min and max of the slice it is
                       showing, and rescales as you move through the volume.
    Using any of the global controls switches back to Global mode.

Command line
    python slice_viewer.py                      # demo volume, 64 x 80 x 96
    python slice_viewer.py --demo 40 120 70     # demo volume of a chosen shape
    python slice_viewer.py volume.npy           # load a .npy file
    python slice_viewer.py data.npz --key vol   # load one array from a .npz file
    python slice_viewer.py volume.npy --cmap gray --clip 1 --names z y x
    python slice_viewer.py volume.npy --per-slice   # start in per-slice mode

From Python / Jupyter
    from slice_viewer import show_slices
    viewer = show_slices(my_array)   # in Jupyter run `%matplotlib widget` first,
                                     # and keep the returned object alive
"""

import argparse
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons, RangeSlider, Slider, TextBox
from mpl_toolkits.axes_grid1 import make_axes_locatable

GLOBAL, PER_SLICE = "Global", "Per slice"


def _finite_range(a):
    """(min, max) of the finite values in a, or () if there are none. Never copies a."""
    finite = np.isfinite(a)
    if finite.all():
        return float(a.min()), float(a.max())
    if not finite.any():
        return ()
    return (float(a.min(where=finite, initial=np.inf)),
            float(a.max(where=finite, initial=-np.inf)))


def _nondegenerate(lo, hi):
    """Widen a zero-width range (e.g. a constant slice) so the colour mapping stays defined."""
    if hi > lo:
        return lo, hi
    pad = 0.5 * (abs(lo) or 1.0)
    return lo - pad, hi + pad


def _fmt(v):
    return f"{v:.6g}"


class SliceViewer:
    # For each panel: (axis held fixed, axis shown horizontally, axis shown vertically)
    PANELS = ((0, 2, 1), (1, 2, 0), (2, 1, 0))

    def __init__(self, volume, cmap="viridis", clip=0.0, vmin=None, vmax=None,
                 per_slice=False, aspect="equal", axis_names=("axis 0", "axis 1", "axis 2"),
                 title=None, crosshair_color="red"):
        vol = np.asarray(volume)
        if vol.ndim != 3:
            raise ValueError(f"Expected a 3D array, got shape {vol.shape}")
        if np.iscomplexobj(vol):
            print("Complex input: displaying the magnitude.", file=sys.stderr)
            vol = np.abs(vol)
        if vol.dtype == bool:
            vol = vol.astype(np.uint8)
        if not 0 <= clip < 50:
            raise ValueError("clip must be in [0, 50)")

        self.vol = vol
        self.shape = vol.shape
        self.names = list(axis_names)
        self.idx = [n // 2 for n in self.shape]
        self.mode = PER_SLICE if per_slice else GLOBAL
        self._ranges = [None] * 3       # cached per-slice (min, max); None = needs recomputing

        # Global colour limits: from vmin/vmax if given, otherwise the [clip, 100-clip] percentiles.
        rng = _finite_range(vol)
        if not rng:
            raise ValueError("Array contains no finite values")
        self.data_range = _nondegenerate(*rng)
        if clip > 0:
            finite = vol[np.isfinite(vol)]      # percentiles need a one-off copy
            lo, hi = (float(q) for q in np.percentile(finite, [clip, 100 - clip]))
            del finite
        else:
            lo, hi = rng
        lo = lo if vmin is None else float(vmin)
        hi = hi if vmax is None else float(vmax)
        if hi < lo:
            raise ValueError("vmax must be greater than vmin")
        self.global_lim = _nondegenerate(lo, hi)

        # ---- figure and image panels, each with its own colorbar ----
        self.fig, axes = plt.subplots(1, 3, figsize=(15, 7))
        self.axes = list(axes)
        self.fig.subplots_adjust(left=0.05, right=0.96, top=0.85, bottom=0.34, wspace=0.45)
        if title:
            self.fig.suptitle(title, fontsize=13)
        self.info = self.fig.text(0.5, 0.91, "", ha="center", fontsize=11, family="monospace")

        self.images, self.vlines, self.hlines = [], [], []
        for ax, (s, xa, ya) in zip(self.axes, self.PANELS):
            im = ax.imshow(self._slice(s), cmap=cmap, vmin=self.global_lim[0], vmax=self.global_lim[1],
                           origin="lower", aspect=aspect, interpolation="nearest")
            ax.set_xlabel(self.names[xa])
            ax.set_ylabel(self.names[ya])
            line_kw = dict(color=crosshair_color, lw=1.3, alpha=0.85)
            self.vlines.append(ax.axvline(self.idx[xa], **line_kw))
            self.hlines.append(ax.axhline(self.idx[ya], **line_kw))
            cax = make_axes_locatable(ax).append_axes("right", size="4%", pad=0.08)
            self.fig.colorbar(im, cax=cax)
            self.images.append(im)

        # ---- index sliders (bottom left) ----
        self.sliders = []
        for a, n in enumerate(self.shape):
            sax = self.fig.add_axes([0.13, 0.22 - 0.06 * a, 0.40, 0.03])
            sl = Slider(sax, f"{self.names[a]}  (N{a + 1}={n})", 0, max(n - 1, 1),
                        valinit=self.idx[a], valstep=1, valfmt="%d", initcolor="none")
            if n == 1:
                sl.set_active(False)
            sl.on_changed(lambda val, a=a: self._set_index(a, val))
            self.sliders.append(sl)

        # ---- colour-limit controls (bottom right) ----
        self.fig.text(0.62, 0.255, "Color limits", fontsize=11, weight="bold")
        rax = self.fig.add_axes([0.615, 0.10, 0.10, 0.12], frameon=False)
        self.radio = RadioButtons(rax, (GLOBAL, PER_SLICE), active=(self.mode == PER_SLICE))
        self.radio.on_clicked(self._on_mode)

        rsax = self.fig.add_axes([0.775, 0.215, 0.195, 0.03])
        self.range_slider = RangeSlider(rsax, "", *self.data_range,
                                        valinit=self._clamp_to_data(*self.global_lim))
        self.range_slider.valtext.set_visible(False)   # the text boxes show the values
        self.range_slider.on_changed(lambda val: self._set_global(*val))

        self.vmin_box = TextBox(self.fig.add_axes([0.775, 0.13, 0.08, 0.045]), "vmin ",
                                initial=_fmt(self.global_lim[0]))
        self.vmax_box = TextBox(self.fig.add_axes([0.89, 0.13, 0.08, 0.045]), "vmax ",
                                initial=_fmt(self.global_lim[1]))
        self.vmin_box.on_submit(lambda text: self._on_box(0, text))
        self.vmax_box.on_submit(lambda text: self._on_box(1, text))

        self.reset_btn = Button(self.fig.add_axes([0.775, 0.06, 0.195, 0.045]), "Data min/max")
        self.reset_btn.on_clicked(lambda event: self._set_global(*self.data_range))

        # ---- mouse / keyboard ----
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        self._style_global_controls()
        self._redraw()

    # ------------------------------------------------------------ slicing
    def _slice(self, axis):
        # Basic indexing returns a view, so this costs nothing whatever the memory layout.
        # (np.take would copy the *entire* array to C order first whenever it isn't
        # C-contiguous, e.g. after a transpose.) The remaining axes keep their order,
        # which matches PANELS (rows = y, cols = x).
        index = [slice(None)] * 3
        index[axis] = self.idx[axis]
        return self.vol[tuple(index)]

    def _clamp(self, axis, value):
        return int(np.clip(round(value), 0, self.shape[axis] - 1))

    def _set_index(self, axis, value):
        self.idx[axis] = self._clamp(axis, value)
        # Only the panel cut along this axis shows new data; the other two just move a crosshair.
        self._update_panel(axis)
        self._update_overlays()
        self.fig.canvas.draw_idle()

    def _goto(self, axis, value):
        # Going through the slider keeps it in sync; its callback triggers the redraw.
        value = self._clamp(axis, value)
        if value != self.idx[axis]:
            self.sliders[axis].set_val(value)

    def _update_panel(self, p):
        # Panel p shows the slice along axis p (see PANELS).
        self.images[p].set_data(self._slice(p))
        # Left-aligned so it never collides with a colorbar offset label (e.g. "1e-11+0.0238").
        self.axes[p].set_title(f"{self.names[p]} = {self.idx[p]}", loc="left")
        self._ranges[p] = None
        self._apply_clim([p])

    def _update_overlays(self):
        for p, (_, xa, ya) in enumerate(self.PANELS):
            self.vlines[p].set_xdata([self.idx[xa]] * 2)
            self.hlines[p].set_ydata([self.idx[ya]] * 2)
        i, j, k = self.idx
        self.info.set_text(f"(i, j, k) = ({i}, {j}, {k})    value = {self.vol[i, j, k]:.5g}")

    def _redraw(self):
        for p in range(3):
            self._update_panel(p)
        self._update_overlays()
        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------ colour limits
    def _apply_clim(self, panels=(0, 1, 2)):
        for p in panels:
            if self.mode == GLOBAL:
                lim = _nondegenerate(*self.global_lim)
            else:
                if self._ranges[p] is None:     # once per slice, then cached
                    self._ranges[p] = _finite_range(self._slice(p))
                if not self._ranges[p]:         # all-NaN slice: keep whatever was there
                    continue
                lim = _nondegenerate(*self._ranges[p])
            self.images[p].set_clim(*lim)   # the panel's colorbar follows automatically

    def _clamp_to_data(self, lo, hi):
        a, b = self.data_range
        return float(np.clip(lo, a, b)), float(np.clip(hi, a, b))

    def _set_global(self, lo, hi):
        self.global_lim = (float(lo), float(hi))
        self._sync_global_widgets()
        if self.mode != GLOBAL:
            self.radio.set_active(0)        # fires _on_mode, which redraws
        else:
            self._apply_clim()
            self.fig.canvas.draw_idle()

    def _sync_global_widgets(self):
        # Update the widgets without re-triggering their callbacks.
        lo, hi = self.global_lim
        for widget, value in ((self.vmin_box, _fmt(lo)), (self.vmax_box, _fmt(hi)),
                              (self.range_slider, self._clamp_to_data(lo, hi))):
            widget.eventson = False
            widget.set_val(value)
            widget.eventson = True

    def _on_box(self, which, text):
        try:
            value = float(text)
        except ValueError:
            value = np.nan
        lim = list(self.global_lim)
        lim[which] = value
        if not np.isfinite(value) or lim[0] >= lim[1]:
            self._sync_global_widgets()     # reject: put the previous value back
            self.fig.canvas.draw_idle()
            return
        self._set_global(*lim)

    def _on_mode(self, label):
        self.mode = label
        self._style_global_controls()
        self._apply_clim()
        self.fig.canvas.draw_idle()

    def _style_global_controls(self):
        # Dim the global controls while they're not in effect (they stay clickable).
        active = self.mode == GLOBAL
        color = "black" if active else "0.6"
        for box in (self.vmin_box, self.vmax_box):
            box.label.set_color(color)
            box.text_disp.set_color(color)
        self.range_slider.poly.set_alpha(1.0 if active else 0.3)

    # ------------------------------------------------------------ event handlers
    def _panel_of(self, event):
        return self.axes.index(event.inaxes) if event.inaxes in self.axes else None

    def _on_click(self, event):
        p = self._panel_of(event)
        toolbar = getattr(self.fig.canvas, "toolbar", None)
        if p is None or event.button != 1 or event.xdata is None or getattr(toolbar, "mode", ""):
            return
        _, xa, ya = self.PANELS[p]
        self._goto(xa, event.xdata)
        self._goto(ya, event.ydata)

    def _on_scroll(self, event):
        p = self._panel_of(event)
        if p is not None:
            s = self.PANELS[p][0]
            self._goto(s, self.idx[s] + (1 if event.button == "up" else -1))

    def _on_key(self, event):
        if self.vmin_box.capturekeystrokes or self.vmax_box.capturekeystrokes:
            return                          # typing in a text box
        steps = {"up": 1, "down": -1, "pageup": 10, "pagedown": -10}
        p = self._panel_of(event)
        if p is not None and event.key in steps:
            s = self.PANELS[p][0]
            self._goto(s, self.idx[s] + steps[event.key])

    def show(self):
        plt.show()


def show_slices(volume, **kwargs):
    """Open the viewer for a 3D array. Returns the viewer; keep a reference to it."""
    viewer = SliceViewer(volume, **kwargs)
    viewer.show()
    return viewer


def demo_volume(n1=64, n2=80, n3=96):
    """A torus around axis 0, an off-centre blob and a gentle ramp along axis 0."""
    x = np.linspace(-1, 1, n1)[:, None, None]
    y = np.linspace(-1, 1, n2)[None, :, None]
    z = np.linspace(-1, 1, n3)[None, None, :]
    torus = np.exp(-((np.hypot(y, z) - 0.55) ** 2 + x ** 2) / (2 * 0.12 ** 2))
    blob = 0.8 * np.exp(-((x - 0.4) ** 2 + (y + 0.3) ** 2 + (z - 0.5) ** 2) / (2 * 0.1 ** 2))
    ramp = 0.15 * (x + 1)
    return torus + blob + ramp


def load_volume(path, key=None):
    if path.endswith(".npz"):
        with np.load(path) as data:
            name = key or data.files[0]
            if name not in data.files:
                raise KeyError(f"'{name}' not in {path}; available: {data.files}")
            return data[name]
    return np.load(path)


def main():
    p = argparse.ArgumentParser(description="Interactive orthogonal slice viewer for a 3D array.")
    p.add_argument("file", nargs="?", help=".npy or .npz file with an N1 x N2 x N3 array (omit for a demo)")
    p.add_argument("--key", help="array name inside a .npz file (default: the first one)")
    p.add_argument("--demo", nargs=3, type=int, metavar=("N1", "N2", "N3"), default=(64, 80, 96),
                   help="shape of the demo volume when no file is given")
    p.add_argument("--cmap", default="viridis", help="matplotlib colormap (default: viridis)")
    p.add_argument("--clip", type=float, default=0.0,
                   help="start the global colour range at the [P, 100-P] percentiles, e.g. 1")
    p.add_argument("--vmin", type=float, help="starting global lower colour limit")
    p.add_argument("--vmax", type=float, help="starting global upper colour limit")
    p.add_argument("--per-slice", action="store_true",
                   help="start with each panel scaled to its own slice's min/max")
    p.add_argument("--aspect", default="equal", help="'equal' for square voxels, 'auto' to fill each panel")
    p.add_argument("--names", nargs=3, metavar=("A0", "A1", "A2"), help="labels for the axes, e.g. z y x")
    args = p.parse_args()

    if args.file:
        vol, title = load_volume(args.file, args.key), args.file
    else:
        vol, title = demo_volume(*args.demo), "Demo volume"
    print(f"Loaded array: shape {vol.shape}, dtype {vol.dtype}")

    show_slices(vol, cmap=args.cmap, clip=args.clip, vmin=args.vmin, vmax=args.vmax,
                per_slice=args.per_slice, aspect=args.aspect,
                axis_names=args.names or ("axis 0", "axis 1", "axis 2"),
                title=f"{title}   shape {vol.shape}")


if __name__ == "__main__":
    main()
