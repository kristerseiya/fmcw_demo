"""
Visualize an N x M complex-valued signal that is sparse in its columns, and
examine how the phase evolves across rows in a chosen set of columns.

Laid out for few rows (N ~ 10) and many columns (M ~ 1000+), but works generally.

Static use:
    from complex_sparse_viz import visualize
    v = visualize(X)                          # bottom row: strongest auto-detected columns
    v = visualize(X, columns=[3, 17, 1024])   # bottom row: your choice of columns
    v.fig.savefig("out.png")

Interactive use (any GUI backend, or `%matplotlib widget` in Jupyter):
    v = visualize(X); plt.show()
    - left-click near a column in any of the top three panels to add/remove it
      (clicks snap to the strongest column within a few pixels; zoom in for precision)
    - right-click to reset to the default selection
    - v.selected is the current selection; v.set_columns([...]) changes it from code
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb, to_rgba
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import offset_copy


# ----------------------------------------------------------------- example data
def make_example(N=10, M=1200, n_active=12, noise=0.05, seed=0):
    """Column-sparse complex matrix; each active column gets a random phase law."""
    rng = np.random.default_rng(seed)
    X = noise * (rng.standard_normal((N, M)) + 1j * rng.standard_normal((N, M))) / np.sqrt(2)
    n = np.arange(N)
    for c in rng.choice(M, n_active, replace=False):
        amp = rng.uniform(0.4, 1.0)
        phi0, rate, curv = rng.uniform(-np.pi, np.pi), rng.uniform(-1.2, 1.2), rng.uniform(-0.06, 0.06)
        X[:, c] += amp * np.exp(1j * (phi0 + rate * n + curv * n**2))
    return X


# --------------------------------------------------------------------- analysis
def find_active_columns(X, k=None, rel_threshold=0.2):
    """Return (sorted active column indices, column L2 norms, threshold or None)."""
    energy = np.linalg.norm(X, axis=0)
    if k is not None:
        return np.sort(np.argsort(energy)[::-1][:k]), energy, None
    thr = rel_threshold * energy.max()
    return np.flatnonzero(energy >= thr), energy, thr


def complex_to_rgb(Z, gamma=1.0):
    """Domain colouring: hue = phase, brightness = (|Z| / max|Z|) ** gamma.
    Linear magnitude (gamma=1) keeps a weak noise floor dark so sparse columns stand out;
    use gamma < 1 to lift weaker content."""
    val = (np.abs(Z) / np.abs(Z).max()) ** gamma
    hue = (np.angle(Z) + np.pi) / (2 * np.pi)
    return hsv_to_rgb(np.stack([hue, np.ones_like(hue), val], axis=-1))


# ----------------------------------------------------------------------- viewer
class ColumnPhaseViewer:
    PICK_TOL_PX = 6          # click-snapping radius in screen pixels

    def __init__(self, X, columns=None, k=None, rel_threshold=0.2, db_range=40.0,
                 max_default=8, figsize=(16, 12)):
        self.X = np.asarray(X)
        N, M = self.X.shape
        self.active, self.energy, self.thr = find_active_columns(self.X, k, rel_threshold)
        strongest = self.active[np.argsort(self.energy[self.active])[::-1]]
        self.default = sorted(int(c) for c in strongest[:max_default])
        self.selected = []
        self._palette = plt.cm.tab10.colors
        self._color_idx = {}
        self._markers = []

        self.fig = fig = plt.figure(figsize=figsize, constrained_layout=True)
        fig._column_viewer = self  # keep the viewer (and its callbacks) alive with the figure
        gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 0.8, 1.5])
        self.ax_mag = fig.add_subplot(gs[0, :])
        self.ax_dom = fig.add_subplot(gs[1, :], sharex=self.ax_mag)
        self.ax_en = fig.add_subplot(gs[2, :], sharex=self.ax_mag)
        self.ax_ph = fig.add_subplot(gs[3, 0])
        self.ax_dp = fig.add_subplot(gs[3, 1])
        self.ax_iq = fig.add_subplot(gs[3, 2])
        self._draw_top(db_range)

        self.set_columns(self.default if columns is None else columns)
        fig.canvas.mpl_connect("button_press_event", self._on_click)

    # ---- static top panels -------------------------------------------------
    def _draw_top(self, db_range):
        X, (N, M) = self.X, self.X.shape
        # sharp pixels when there is room; "antialiased" when columns outnumber screen pixels,
        # so a single strong column is never dropped by nearest-neighbour resampling
        px_per_col = 0.85 * self.fig.get_figwidth() * self.fig.dpi / M
        img_kw = dict(aspect="auto", interpolation="nearest" if px_per_col >= 2 else "antialiased")

        ax = self.ax_mag
        mag_db = 20 * np.log10(np.abs(X) / np.abs(X).max() + 1e-12)
        im = ax.imshow(mag_db, cmap="magma", vmin=-db_range, vmax=0, **img_kw)
        self.fig.colorbar(im, cax=ax.inset_axes([1.01, 0, 0.012, 1]), label="|X| (dB re max)")
        ax.set_title("Magnitude", pad=16)
        ax.set_ylabel("row")

        ax = self.ax_dom
        ax.imshow(complex_to_rgb(X), **img_kw)
        sm = plt.cm.ScalarMappable(norm=plt.Normalize(-np.pi, np.pi), cmap="hsv")
        cb = self.fig.colorbar(sm, cax=ax.inset_axes([1.01, 0, 0.012, 1]), label="phase (rad)")
        cb.set_ticks([-np.pi, 0, np.pi])
        cb.set_ticklabels(["-π", "0", "π"])
        ax.set_title("Phase (hue) × magnitude (brightness)", pad=16)
        ax.set_ylabel("row")

        ax = self.ax_en
        is_act = np.zeros(M, bool)
        is_act[self.active] = True
        colors = np.where(is_act[:, None], to_rgba("0.2"), to_rgba("0.75"))
        ax.vlines(np.arange(M), 0, self.energy, colors=colors, lw=float(np.clip(800 / M, 0.8, 6)))
        if self.thr is not None:
            ax.axhline(self.thr, ls="--", c="k", lw=1)
            ax.text(0.003, self.thr, "detection threshold", transform=ax.get_yaxis_transform(),
                    va="bottom", fontsize=8)
        ax.set(xlabel="column", ylabel="||X[:, m]||")

        for ax in (self.ax_mag, self.ax_dom):
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax.tick_params(labelbottom=False)
        ax = self.ax_en
        ax.set_xlim(-0.5, M - 0.5)
        ax.set_ylim(0, 1.12 * self.energy.max())
        for ax in (self.ax_mag, self.ax_dom, self.ax_en):
            ax.set_autoscale_on(False)   # markers added later must not change the limits

    # ---- selection -----------------------------------------------------------
    def set_columns(self, columns):
        M = self.X.shape[1]
        cols = sorted({int(c) for c in columns})
        bad = [c for c in cols if not 0 <= c < M]
        if bad:
            raise IndexError(f"column(s) {bad} out of range 0..{M - 1}")
        # stable colours: a column keeps its colour while selected; new ones take a free slot
        self._color_idx = {c: i for c, i in self._color_idx.items() if c in cols}
        for c in cols:
            if c not in self._color_idx:
                used = set(self._color_idx.values())
                free = [i for i in range(len(self._palette)) if i not in used]
                self._color_idx[c] = free[0] if free else len(self._color_idx) % len(self._palette)
        self.selected = cols
        self._update_markers()
        self._update_title()
        self._draw_bottom()
        self.fig.canvas.draw_idle()

    def _color(self, c):
        return self._palette[self._color_idx[c]]

    def _update_markers(self):
        for artist in self._markers:
            artist.remove()
        self._markers = []
        sel = self.selected
        if not sel:
            return
        cols = [self._color(c) for c in sel]
        style = dict(c=cols, edgecolors="k", linewidths=0.5, zorder=5)
        for ax in (self.ax_mag, self.ax_dom):
            trans = offset_copy(ax.get_xaxis_transform(), fig=self.fig, y=7, units="points")
            self._markers.append(ax.scatter(sel, np.ones(len(sel)), transform=trans, marker="v",
                                            s=70, clip_on=False, **style))
        self._markers.append(self.ax_en.scatter(sel, self.energy[sel], s=45, **style))

    def _update_title(self):
        M = self.X.shape[1]
        frac = (self.energy[self.active] ** 2).sum() / (self.energy ** 2).sum()
        shown = ", ".join(map(str, self.selected[:20])) + (" …" if len(self.selected) > 20 else "")
        self.ax_en.set_title(
            f"Column L2 norm — {len(self.active)} of {M} columns flagged active, holding "
            f"{frac:.1%} of the energy\nbottom row: [{shown}]   "
            f"(left-click: add/remove column, right-click: reset)", fontsize=10)

    def _on_click(self, event):
        if event.inaxes not in (self.ax_mag, self.ax_dom, self.ax_en) or event.xdata is None:
            return
        toolbar = getattr(self.fig.canvas, "toolbar", None)
        if toolbar is not None and getattr(toolbar, "mode", "") != "":
            return  # zoom/pan tool is active: leave the click to it
        if event.button == 3:
            self.set_columns(self.default)
            return
        if event.button != 1:
            return
        ax, x = event.inaxes, event.xdata
        x0, x1 = ax.get_xlim()
        tol = max(self.PICK_TOL_PX * abs(x1 - x0) / max(ax.bbox.width, 1), 0.5)
        sel = np.array(self.selected, dtype=int)
        near = sel[np.abs(sel - x) <= tol]
        if near.size:                                    # clicked on a selected column: remove it
            c = int(near[np.argmin(np.abs(near - x))])
            self.set_columns([s for s in self.selected if s != c])
        else:                                            # add the strongest column nearby
            M = self.X.shape[1]
            lo, hi = max(int(np.ceil(x - tol)), 0), min(int(np.floor(x + tol)), M - 1)
            if lo <= hi:
                self.set_columns(self.selected + [lo + int(np.argmax(self.energy[lo:hi + 1]))])

    # ---- bottom row: phase across rows for the selected columns --------------
    def _draw_bottom(self):
        N = self.X.shape[0]
        rows = np.arange(N)
        sel = self.selected
        axes = (self.ax_ph, self.ax_dp, self.ax_iq)
        for ax in axes:
            ax.clear()
        self.ax_ph.set(title="Unwrapped phase (relative to row 0)", xlabel="row", ylabel="phase (rad)")
        self.ax_dp.set(title="Phase step  Δφ[n] = ∠( x[n] · conj(x[n−1]) )",
                       xlabel="row n", ylabel="Δφ (rad)")
        self.ax_iq.set(title="IQ trajectory (RMS-normalised;  ● row 0,  ■ last row)",
                       xlabel="Re", ylabel="Im")
        for ax in axes:
            ax.grid(alpha=0.3)
        if not sel:
            for ax in axes:
                ax.text(0.5, 0.5, "no columns selected", transform=ax.transAxes,
                        ha="center", va="center", color="0.5")
            return

        Xs = self.X[:, sel]
        phase = np.unwrap(np.angle(Xs), axis=0)
        phase -= phase[0]
        prod = Xs[1:] * np.conj(Xs[:-1])
        dphi = np.angle(prod)                      # wrap-safe, no unwrapping needed
        rate = np.angle(prod.sum(axis=0))          # magnitude-weighted average rate
        pts = dict(marker="o", ms=4) if N <= 60 else {}

        t = np.linspace(0, 2 * np.pi, 200)
        self.ax_iq.plot(np.cos(t), np.sin(t), ls=":", c="0.6", lw=1)
        for j, c in enumerate(sel):
            col = self._color(c)
            self.ax_ph.plot(rows, phase[:, j], color=col, label=f"col {c}", **pts)
            self.ax_dp.plot(rows[1:], dphi[:, j], color=col,
                            label=f"col {c}: {rate[j]:+.3f} rad/row", **pts)
            z = Xs[:, j] / max(np.sqrt(np.mean(np.abs(Xs[:, j]) ** 2)), 1e-300)
            self.ax_iq.plot(z.real, z.imag, color=col, lw=1, alpha=0.85, **pts)
            self.ax_iq.plot(z.real[0], z.imag[0], "o", ms=8, color=col, mec="k", mew=0.6)
            self.ax_iq.plot(z.real[-1], z.imag[-1], "s", ms=7, color=col, mec="k", mew=0.6)

        self.ax_dp.axhline(0, c="k", lw=0.5)
        for ax in (self.ax_ph, self.ax_dp):
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            if len(sel) <= 12:
                ax.legend(fontsize=8, ncol=1 if len(sel) <= 6 else 2)
        self.ax_iq.set_aspect("equal", adjustable="datalim")


def visualize(X, columns=None, **kwargs):
    """columns=None -> strongest auto-detected columns; otherwise an iterable of column indices.
    Other kwargs: k, rel_threshold, db_range, max_default, figsize."""
    return ColumnPhaseViewer(X, columns=columns, **kwargs)


if __name__ == "__main__":
    X = make_example()
    v = visualize(X)
    v.fig.savefig("complex_sparse_viz.png", dpi=120)
    plt.show()
