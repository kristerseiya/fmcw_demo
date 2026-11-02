"""
parallel_runner.py -- generic machinery; you should not need to edit this.

run_windows(x, win, hop, func, meta=...) slides a window along the rows of x
(an N x K array; any array works, windows are taken along axis 0):

    window 0 : x[0     : win,         :]
    window 1 : x[hop   : hop + win,   :]
    window j : x[j*hop : j*hop + win, :]

Windows may overlap (hop < win); only full windows are used. Each window is
processed in a worker process by

    func(segment, seg_idx, meta, **params) -> dict of scalars / arrays

per_window={name: rows} additionally passes row j of `rows` to window j as
keyword `name` (e.g. results of a previous run, one row per window). `rows`
is an array, or a dict of arrays -> window j gets {key: array[j]}.

save_results(...) stacks every key of those dicts across windows (axis 0 =
window, sorted by start row) and writes one .npz file.

x is put in shared memory once, so workers read it without copying it (a
multi-GB array is fine). Every call gets its own copy of its segment, so the
algorithm may modify `segment` in place without affecting other windows.
"""
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from multiprocessing import shared_memory

import numpy as np

# Worker-process globals (set once per worker by _init_worker).
_X = None
_SHM = None
_WIN = None
_FUNC = None
_META = None
_PARAMS = None


_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")


@contextmanager
def _limited_threads(n):
    # numpy/scipy BLAS threads x many processes = oversubscription -> slow.
    # Set in the parent so spawned workers inherit it before importing numpy.
    old = {v: os.environ.get(v) for v in _THREAD_VARS}
    os.environ.update({v: str(n) for v in _THREAD_VARS})
    try:
        yield
    finally:
        for v, val in old.items():
            if val is None:
                os.environ.pop(v, None)
            else:
                os.environ[v] = val


def _init_worker(x, win, func, meta, params):
    """x is either the array itself (serial) or (shm_name, shape, dtype)."""
    global _X, _SHM, _WIN, _FUNC, _META, _PARAMS
    if isinstance(x, tuple):
        name, shape, dtype = x
        _SHM = shared_memory.SharedMemory(name=name)   # keep alive (global)
        _X = np.ndarray(shape, dtype=dtype, buffer=_SHM.buf)
    else:
        _X = x
    _WIN, _FUNC, _META, _PARAMS = win, func, meta, params


def _run_batch(jobs):
    """jobs: list of (seg_idx, start, per-window kwargs).
    Returns [(seg_idx, output, error_text)]."""
    out = []
    for j, s, kw in jobs:
        seg = _X[s:s + _WIN].copy()
        try:
            out.append((j, _FUNC(seg, j, _META, **_PARAMS, **kw), None))
        except Exception:
            out.append((j, None, traceback.format_exc()))
    return out


class WindowError(RuntimeError):
    pass


def window_starts(n_rows, win, hop, start=0, stop=None, max_windows=None):
    """Start row of every full window x[s:s+win] that fits in rows [start, stop)."""
    stop = n_rows if stop is None else min(stop, n_rows)
    starts = np.arange(start, stop - win + 1, hop)
    if max_windows is not None:
        starts = starts[:max_windows]
    return starts


def run_windows(x, win, hop, func, meta=None, params=None, per_window=None,
                n_workers=None, start=0, stop=None, max_windows=None,
                on_error="raise", chunksize=1, blas_threads=1, verbose=True):
    """
    x           : array, windows are taken along axis 0 (e.g. N x K)
    win, hop    : window length and shift, in rows (e.g. 3*L and L)
    func        : func(segment, seg_idx, meta, **params) -> dict
                  (must be a top-level function in an importable module)
                  segment = x[start_j : start_j + win] (a copy),
                  seg_idx = j (0, 1, 2, ...), start_j = start + j*hop
    meta        : dict describing the whole array; passed to every call
    params      : dict of extra keyword arguments for func (optional)
    per_window  : dict {name: rows} (optional); rows is an array with one row
                  per window in rows [start, stop) (axis 0, before
                  max_windows) and window j gets func(..., name=rows[j]); or
                  rows is a dict of such arrays and window j gets
                  func(..., name={key: rows[key][j]}). Only row j is sent to
                  the worker. E.g. dict(prev_result=results_of_previous_run)
    n_workers   : number of processes; None = all CPUs; 1 = run serially in
                  this process (use this for debugging / breakpoints)
    start, stop : only use rows [start, stop) of x
    max_windows : only run the first max_windows windows (quick tests)
    on_error    : 'raise' -> stop at the first window that raises
                  'skip'  -> report it, continue; its outputs become NaN
    chunksize   : windows sent to a worker per task (1 is best unless func
                  takes only ~milliseconds)
    blas_threads: numpy/BLAS threads per worker (1 avoids oversubscription)

    Returns (starts, outputs, errors):
      starts  : np.ndarray, start row of each window
      outputs : list of dicts of np.ndarrays (None where the window failed)
      errors  : dict {seg_idx: traceback text} of failed windows
    """
    if on_error not in ("raise", "skip"):
        raise ValueError("on_error must be 'raise' or 'skip'")
    x = np.asarray(x)
    meta = dict(meta or {})
    params = dict(params or {})
    per_window = dict(per_window or {})
    starts = window_starts(len(x), win, hop, start, stop)
    for k, v in per_window.items():
        _check_rows(f"per_window['{k}']", v, len(starts),
                    f"[{start}, {len(x) if stop is None else stop})")
    if max_windows is not None:
        starts = starts[:max_windows]
    n = len(starts)
    if n == 0:
        raise ValueError(f"no full window of {win} rows fits in rows "
                         f"[{start}, {len(x) if stop is None else stop})")
    if n_workers is None:
        n_workers = os.cpu_count()
    n_workers = max(1, min(n_workers, n))
    jobs = [(j, s, {k: _row(v, j) for k, v in per_window.items()})
            for j, s in enumerate(starts.tolist())]
    batches = [jobs[i:i + chunksize] for i in range(0, n, chunksize)]

    if verbose:
        print(f"[runner] x {x.shape} {x.dtype}: {n} windows of {win} rows, hop {hop}, "
              f"rows {starts[0]}..{starts[-1] + win}, {n_workers} worker(s)")
    outputs = [None] * n
    errors = {}
    bar = _ProgressBar(n, verbose)

    def handle(j, out, err):
        if err is None:
            outputs[j] = _as_record(out, j)
            return
        errors[j] = err
        where = f"window {j} (rows {starts[j]}:{starts[j] + win})"
        if on_error == "raise":
            raise WindowError(f"{func.__name__} failed on {where}:\n{err}")
        if len(errors) == 1:
            bar.write(f"[runner] {where} failed (on_error='skip', continuing):\n{err}")
        else:
            bar.write(f"[runner] {where} failed: {err.strip().splitlines()[-1]}")

    t0 = time.perf_counter()
    try:
        if n_workers == 1:
            _init_worker(x, win, func, meta, params)
            for batch in batches:
                for res in _run_batch(batch):
                    handle(*res)
                bar.update(len(batch))
        else:
            _run_parallel(x, win, func, meta, params, n_workers, blas_threads,
                          batches, handle, bar)
    finally:
        bar.close()

    if verbose:
        dt = time.perf_counter() - t0
        print(f"[runner] done in {dt:.1f} s ({dt / n * n_workers:.3f} s per window "
              f"per worker); {n - len(errors)} ok, {len(errors)} failed")
    return starts, outputs, errors


def _run_parallel(x, win, func, meta, params, n_workers, blas_threads,
                  batches, handle, bar):
    shm = shared_memory.SharedMemory(create=True, size=max(1, x.nbytes))
    try:
        shared = np.ndarray(x.shape, dtype=x.dtype, buffer=shm.buf)
        shared[...] = x
        del shared   # drop the view, otherwise shm.close() fails
        with _limited_threads(blas_threads), ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_worker,
            initargs=((shm.name, x.shape, x.dtype.str), win, func, meta, params),
        ) as ex:
            futs = [ex.submit(_run_batch, b) for b in batches]
            try:
                for fut in as_completed(futs):
                    res = fut.result()
                    for r in res:
                        handle(*r)
                    bar.update(len(res))
            except BaseException:
                for f in futs:
                    f.cancel()
                raise
    finally:
        shm.close()
        shm.unlink()


def _check_rows(name, rows, n, where):
    """rows (an array, or a dict of arrays) must have n rows along axis 0."""
    if isinstance(rows, dict):
        for k, v in rows.items():
            _check_rows(f"{name}['{k}']", v, n, where)
        return
    try:
        m = len(rows)
    except TypeError:          # scalar / 0-d array
        m = "no"
    if m != n:
        raise ValueError(f"{name} has {m} rows (axis 0), but rows {where} "
                         f"give {n} windows")


def _row(rows, j):
    """Row j of an array, or {key: row j} of a dict of arrays."""
    if isinstance(rows, dict):
        return {k: _row(v, j) for k, v in rows.items()}
    return rows[j]


def _as_record(out, j):
    """Check one window's output and convert its values to np.ndarrays."""
    if not isinstance(out, dict):
        raise TypeError(
            f"process_segment must return a dict of scalars/arrays, got "
            f"{type(out).__name__} (window {j}); e.g. return dict(t=t, d=d, s=s)")
    rec = {}
    for k, v in out.items():
        if v is None:          # None = missing for this window -> NaN
            continue
        a = np.asarray(v)
        if a.dtype == object:
            raise TypeError(f"output '{k}' of window {j} is not a scalar or a "
                            f"regular (non-ragged) array: {type(v).__name__}")
        rec[str(k)] = a
    return rec


# --------------------------------------------------------------------------
# collecting / saving
# --------------------------------------------------------------------------

def collect(starts, outputs):
    """
    Sort windows by start row and stack each output key across windows.

    Returns (start_idx, ok, data):
      start_idx : start row of each window (sorted)
      ok        : bool per window, False where the window failed
      data      : {key: array}, axis 0 = window. Scalars become a vector,
                  arrays of shape S become shape (n_windows, *S).
                  Windows that failed (or did not return the key) are NaN;
                  arrays whose length varies between windows are NaN-padded
                  to the largest shape.
    """
    order = np.argsort(starts, kind="stable")
    starts = np.asarray(starts)[order]
    outputs = [outputs[i] for i in order]
    ok = np.array([o is not None for o in outputs])
    keys = list(dict.fromkeys(k for o in outputs if o for k in o))
    data = {k: _stack(k, [o.get(k) if o else None for o in outputs]) for k in keys}
    return starts, ok, data


def _stack(key, vals):
    present = [v for v in vals if v is not None]
    shapes = sorted({v.shape for v in present})
    if len(shapes) == 1 and len(present) == len(vals):
        return np.stack(vals)
    if len({len(s) for s in shapes}) > 1:
        raise ValueError(f"output '{key}' changes number of dimensions "
                         f"between windows: {shapes}")
    shape = tuple(int(d) for d in np.max(shapes, axis=0)) if shapes[0] else ()
    dtype = np.result_type(*[v.dtype for v in present], np.float64)
    if dtype.kind not in "fc":
        raise ValueError(f"output '{key}' ({dtype}) is missing in some windows "
                         f"or varies in shape, and cannot be NaN-padded")
    if len(shapes) > 1:
        print(f"[runner] note: '{key}' has different shapes across windows "
              f"({shapes[0]} ... {shapes[-1]}); NaN-padded to {shape}")
    arr = np.full((len(vals),) + shape, np.nan, dtype=dtype)
    for i, v in enumerate(vals):
        if v is not None:
            arr[(i,) + tuple(slice(0, d) for d in v.shape)] = v
    return arr


def save_results(path, starts, outputs, errors=None, meta=None, **extra):
    """
    Write one .npz with:
      start_idx, ok        -- see collect()
      <key> for each key returned by process_segment (stacked, see collect())
      meta_<key>           -- meta values that are numbers / arrays / strings
      any **extra          -- e.g. L=..., win=..., hop=...
    If errors is non-empty, their tracebacks go to <path>_errors.txt.
    Returns the data dict that was saved.
    """
    start_idx, ok, data = collect(starts, outputs)
    info = dict(start_idx=start_idx, ok=ok)
    info.update({k: np.asarray(v) for k, v in extra.items()})
    for k, v in (meta or {}).items():
        try:
            a = np.asarray(v)
        except Exception:
            continue
        if a.dtype != object:      # skip things like sparse matrices, dicts
            info[f"config_{k}"] = a
    for k in list(data):
        if k in info:
            print(f"[runner] output key '{k}' is reserved; saved as 'out_{k}'")
            data[f"out_{k}"] = data.pop(k)
    data = {**info, **data}
    np.savez(path, **data)
    print(f"[runner] saved {path}  (keys: {', '.join(data)})")

    if errors:
        err_path = os.path.splitext(path)[0] + "_errors.txt"
        with open(err_path, "w") as f:
            for j in sorted(errors):
                f.write(f"===== window {j} (start row {starts[j]}) =====\n{errors[j]}\n")
        print(f"[runner] {len(errors)} failed windows; tracebacks in {err_path}")
    return data


class _ProgressBar:
    """tqdm if available, otherwise a line every ~10%."""

    def __init__(self, total, verbose):
        self.total, self.done, self.verbose, self.tqdm = total, 0, verbose, None
        if verbose:
            try:
                from tqdm import tqdm
                self.tqdm = tqdm(total=total)
            except ImportError:
                pass

    def update(self, k):
        self.done += k
        if self.tqdm is not None:
            self.tqdm.update(k)
        elif self.verbose and (self.done * 10 // self.total
                               > (self.done - k) * 10 // self.total):
            print(f"[runner] {self.done}/{self.total} windows")

    def write(self, msg):
        if self.tqdm is not None:
            self.tqdm.write(msg)
        else:
            print(msg)

    def close(self):
        if self.tqdm is not None:
            self.tqdm.close()
