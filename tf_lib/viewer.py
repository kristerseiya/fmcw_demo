
import numpy as np

from .slice_viewer import SliceViewer

DYN_RANGE_DB = 50.0        # displayed dynamic range

# ---------------------------------------------------------------- display helpers
def to_db(power, dyn_range=DYN_RANGE_DB):
    """10 log10(power / max), floored at -dyn_range."""
    p = np.asarray(power, dtype=np.float64)
    db = 10 * np.log10(p / max(p.max(), 1e-300) + 1e-30)
    return np.maximum(db, -dyn_range).astype(np.float32)


def image(ax, t, f, Z, title):
    """Z indexed [time, frequency]."""
    ax.imshow(Z.T, origin="lower", aspect="auto", cmap="magma", interpolation="nearest",
              extent=[t[0], t[-1], f[0], f[-1]])
    ax.set(title=title, xlabel="time (s)", ylabel="frequency (Hz)")


def overlay_if(ax, tf, inst_freq):
    """Thin dashed IF curves (wrapped; |.| for real data) on a time-frequency image."""
    fs = tf["fs"]
    t = np.arange(inst_freq.shape[1]) / fs
    limits = ax.get_xlim(), ax.get_ylim()
    for f in inst_freq:
        fw = (f + fs / 2) % fs - fs / 2
        fw = np.abs(fw) if tf["real"] else fw
        fw[1:][np.abs(np.diff(fw)) > fs / 4] = np.nan          # no line across a wrap
        ax.plot(t, fw, "--", color="cyan", lw=0.8, alpha=0.8)
    ax.set_xlim(limits[0]); ax.set_ylim(limits[1])


def index_map(sym, label, values, unit):
    """'label = start + step * sym unit' for a uniform grid (the viewer works in indices)."""
    step = values[1] - values[0] if len(values) > 1 else 0.0
    return f"{label} = {values[0]:.4g} + {step:.4g}·{sym} {unit}"


def view_3d(power, tf, f_key, c_key, title, db=True):
    """Open slice_viewer on a (T, F, C) power volume, shown as (chirp, frequency, time).
    db=True: 10 log10(power / max), floored at -DYN_RANGE_DB; db=False: the values as
    they are (colour limits start at the data min / max)."""
    vol = (to_db(power) if db else np.asarray(power, dtype=np.float32)).transpose(2, 1, 0)
    mapping = "    ".join((index_map("i", "chirp rate", tf[c_key], "Hz/s"),
                           index_map("j", "frequency", tf[f_key], "Hz"),
                           index_map("k", "time", tf["t"], "s")))
    limits = dict(vmin=-DYN_RANGE_DB, vmax=0) if db else {}
    return SliceViewer(vol, cmap="magma", aspect="auto",
                       axis_names=("chirp rate", "frequency", "time"),
                       title=f"{title} ({'dB' if db else 'linear'})\n{mapping}", **limits)

