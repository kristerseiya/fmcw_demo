# Zero-crossing hints for the two-sided FCRM ridge

How `tf_lib.chirplet_ridge.estimate_ifs(..., zero_crossings=...)` uses known
locations of IF sign changes to estimate the **signed** IF of a **real**
signal, and how `tf_demo/demo_real_signed_hint.py` exercises it. The
equations are numbered (1)–(24) and follow the code in
`src/tf_lib/chirplet_ridge.py` and `src/tf_lib/if_lib.py`.

## Quick start

The demo's default run (B = 1 Hz, SNR 10 dB, seed 5, hints at the true sign
changes ± 50 ms) in a few lines:

```python
import numpy as np
import tf_lib.if_lib as L

fs, n, B, snr, seed = 1000.0, 4000, 1.0, 10.0, 5
rng = np.random.default_rng(seed)
f = L.if_real(rng, n, fs, B)                                    # signed true IF (Hz)
x = L.add_noise(np.cos(L.phase_from_if(f, fs, rng.uniform(0, 2 * np.pi))),
                L.unit_noise(rng, n, False), snr)               # real signal

zc = np.nonzero(np.diff(np.sign(f)) != 0)[0]                    # true sign changes (samples)
hints = [(c - 50, c + 50) for c in zc]                          # +/- 50 ms at fs = 1 kHz

t, IF = L.fcrm_estimate(x, fs, B, f_scale=0.4 * fs / 2, onesided=False, n_omega_out=512,
                        zero_crossings=hints)

_, IF = L.best_global_sign(t, IF, f, fs)                        # oracle global sign
_, e = L.errors(t, IF, f, fs)
print(f"RMSE {np.sqrt(np.mean(e ** 2)):.1f} Hz")                # 3.1 Hz (177.1 Hz without hints)
```

All other settings are the defaults that the demo uses (`sigma=8`,
`lam_f=0.05`, `lam_c=0.3`, `mode="eq9"`, `prediction=True`,
`hold_sign=True`, `lam_c_cross=0`). `fcrm_estimate` sizes the chirp-rate grid
and the jump limits from the GP bandwidth `B` (eq. 21). For your own data,
without that prior, call the core function directly:

```python
import tf_lib.chirplet_ridge as CRM
t, IF, chirp = CRM.estimate_ifs(x, fs, zero_crossings=[(idx1, idx2), ...])   # IF[0] in Hz
```

`onesided` defaults to `False` when `zero_crossings` is given.

The plain `± 50` intervals above work for seed 5 because its crossings are far
apart. When crossings can be closer than twice the margin, the intervals
would overlap, which is an error (eq. 8). The demo's `hint_intervals()` pulls
each end in to the midpoint between neighbouring centres instead (eq. 22).

Run the demo itself with `python tf_demo/demo_real_signed_hint.py` (add
`--save out.png` for no window).

## The problem

The two-sided spectrum of a real signal holds both `+IF` and `−IF`. Where the
true IF changes sign, these two mirror images cross at 0 Hz. A ridge tracker
on the two-sided FCRM can go wrong in two ways:

* **Bounce:** at a true crossing, it switches to the mirror image and stays
  on one side.
* **False crossing:** where the IF only comes near 0 Hz and turns back, it
  crosses.

Either way the ridge follows the wrong image from that point on. A hint is a
sample interval `[idx1, idx2]` that contains **exactly one** sign change of
the IF.

## Pipeline

| step | function (`src/tf_lib/chirplet_ridge.py`) | output | equations |
|---|---|---|---|
| 1. chirplet transform + frequency/chirp-rate reassignment | `chirplet_reassign` | FCRM volume `R[t, k, b]` | (1) |
| 2. energy | `_energy_log` | `E[t, k, b]` | (2) |
| 3. hints → frames | `_flip_frames` | frame pairs `(t1, t2)` | (7)–(9) |
| 4. coarse DP, then fine DP | `ridge_dp` → `_viterbi_flips` | ridge path `(k_t, b_t)` | (3)–(5), (10)–(18) |
| 5. read-out | `readout` | IF (Hz), chirp rate (Hz/s) | (19) |

Hints only change steps 3 and 4. `estimate_ifs` → `extract_ridges` → `ridge_dp`
passes them through; `if_lib.fcrm_estimate` forwards them, with `lam_c_cross`
as its name for `lam_b_cross`.

## Notation

| symbol | meaning | code |
|---|---|---|
| $t = 0,\dots,T-1$ | frame index, centre at sample $n_t = t\,h$ | `tfr["t"]` |
| $h$ | hop (samples) | `hop` |
| $k = 0,\dots,K-1$ | frequency bin, $\omega_k = -\pi + k\,\Delta\omega$, $\Delta\omega = 2\pi/K$ (rad/sample) | `omega_out`, `d_om` |
| $b = 0,\dots,B-1$ | chirp-rate bin, $\beta_b = -\beta_{\max} + b\,\Delta\beta$, $\Delta\beta = 2\beta_{\max}/(B-1)$ | `beta_out` |
| $q_t = (k_t, b_t)$ | ridge state at frame $t$ | `kp`, `bp` |
| $\lambda_k, \lambda_b$ | frequency / chirp-rate jump penalties (per bin²) | `lam_k`, `lam_b` (`lam_f`, `lam_c` in `fcrm_estimate`) |
| $\lambda_b^{\times}$ | chirp-rate penalty inside hint intervals | `lam_b_cross` (`lam_c_cross`) |
| $D_k, D_b$ | hard jump limits (bins) | `max_dk`, `max_db` |
| $r_k, r_b$ | coarse block size (bins) | `r_k`, `r_b` |
| $\sigma(k)$ | side of 0 Hz of bin $k$, eq. (6) | `side` |
| $c \in \{0,1\}$ | state copy inside an interval, $\varsigma_0 = +1$, $\varsigma_1 = -1$ | copy axis of `C` |

$[\,\cdot\,]$ is the Iverson bracket (1 if true, 0 if false), with
$\infty\cdot 0 = 0$, so $\infty\cdot[\text{cond}]$ adds $+\infty$ exactly when
the condition holds. A maximum over an empty set is $-\infty$.

## FCRM volume and energy

The chirplet transform with a Gaussian window $g$, and its reassignment
operators $\hat\omega$, $\hat\beta$ (selected by `mode`, see the module
docstring), give the FCRM volume. Each transform cell moves its energy to the
output cell that contains its reassigned point, if it is above the threshold
$\gamma$:

$$
C(t,\omega,\beta) = \sum_\tau x[n_t+\tau]\,g(\tau)\,e^{-j\beta\tau^2/2}\,e^{-j\omega\tau},
\qquad
R[t,k,b] = \sum_{\substack{(\omega,\beta):\ \hat\omega \in \text{bin } k,\ \hat\beta \in \text{bin } b \\ \lvert C\rvert^2 \,\ge\, \gamma \max_{\omega,\beta}\lvert C(t,\cdot)\rvert^2}} \lvert C(t,\omega,\beta)\rvert^2
\tag{1}
$$

The same sum weighted by $\hat\omega$ (or $\hat\beta$) gives $R_\omega$
(`Rw_omega`) and $R_\beta$ (`Rw_beta`), which the read-out uses (eq. 19).

The DP works on the log energy. $\tilde R$ is $R$ blurred by a Gaussian with
a standard deviation of 1 bin along $k$ and $b$ (`blur=(1, 1)`), floored at
$10^{-6}$ of the frame maximum, and normalized so that each frame's maximum
is 0:

$$
E[t,k,b] = \log\bigl(\tilde R[t,k,b] + \epsilon_t\bigr) - \max_{k',b'} \log\bigl(\tilde R[t,k',b'] + \epsilon_t\bigr),
\qquad \epsilon_t = 10^{-6}\max_{k,b}\tilde R[t,k,b]
\tag{2}
$$

## The ridge DP without hints

The penalty for moving from $q' = (k', b')$ at frame $t-1$ to $q = (k, b)$ at
frame $t$ is

$$
P_t(q \mid q') = \lambda_k\bigl(k-k'-s(b',b)\bigr)^2 + \lambda_b\,(b-b')^2
+ \infty\cdot\bigl[\,\lvert k-k'\rvert > D_k \ \lor\ \lvert b-b'\rvert > D_b\,\bigr]
\tag{3}
$$

with the chirp-rate prediction (in frequency bins per hop)

$$
s(b',b) = \tfrac12\bigl(\mathrm{sh}_{b'} + \mathrm{sh}_b\bigr),\qquad
\mathrm{sh}_b = \frac{\beta_b\,h}{\Delta\omega}\quad(\texttt{prediction=True}),\qquad \mathrm{sh}_b = 0 \text{ otherwise}.
$$

The ridge maximizes

$$
J(q_{0:T-1}) = \sum_{t=0}^{T-1} E[t,q_t] - \sum_{t=1}^{T-1} P_t(q_t \mid q_{t-1}).
\tag{4}
$$

Viterbi computes this exactly:

$$
V_0(q) = E[0,q],\qquad
V_t(q) = E[t,q] + \max_{q'}\bigl[V_{t-1}(q') - P_t(q\mid q')\bigr],\qquad
\psi_t(q) = \operatorname*{arg\,max}_{q'}\bigl[V_{t-1}(q') - P_t(q\mid q')\bigr],
\tag{5}
$$

then backtracks: $q^\star_{T-1} = \operatorname{arg\,max}_q V_{T-1}(q)$ and
$q^\star_{t-1} = \psi_t(q^\star_t)$. Searching every state is too expensive,
so the DP runs twice: a coarse pass on blocks of $r_k \times r_b$ cells, then
a fine pass in a corridor around the coarse path (eqs. 15–18).

## How the sign changes are enforced

### 1. Side of a state

$$
\sigma(k) = \begin{cases} +1 & \omega_k \ge 0 \\ -1 & \omega_k < 0 \end{cases}
\qquad\Longleftrightarrow\qquad \sigma(k) = +1 \iff k \ge K/2 .
\tag{6}
$$

0 Hz (bin $K/2$) counts as positive. The DP never wraps $k$ around $\pm\pi$
($k - k'$ in eq. 3 is a plain difference), so a path can change side only by
passing 0 Hz.

### 2. From samples to frames (`_flip_frames`)

The hint pairs are sorted by start, and $\mathrm{idx}_1 > \mathrm{idx}_2$ is
swapped. Interval $j$ maps to the nearest frames, adjusted so that
$t_1 \le T-2$ and $t_2 > t_1$:

$$
t_1^{(j)} = \min\Bigl(\operatorname*{arg\,min}_t \bigl\lvert n_t - \mathrm{idx}_1^{(j)}\bigr\rvert,\ T-2\Bigr),
\qquad
t_2^{(j)} = \max\Bigl(\operatorname*{arg\,min}_t \bigl\lvert n_t - \mathrm{idx}_2^{(j)}\bigr\rvert,\ t_1^{(j)}+1\Bigr).
\tag{7}
$$

Neighbouring intervals may share an end frame but must not overlap;
otherwise `_flip_frames` raises a `ValueError`:

$$
t_1^{(j+1)} \ge t_2^{(j)} .
\tag{8}
$$

Two index sets follow:

$$
\mathcal I = \bigcup_j \bigl\{t : t_1^{(j)} < t < t_2^{(j)}\bigr\}\ \ \text{(frames strictly inside)},
\qquad
\mathcal F = \bigcup_j \bigl\{t : t_1^{(j)} < t \le t_2^{(j)}\bigr\}\ \ \text{(moves } t-1\to t \text{ inside)}.
\tag{9}
$$

In code, $\mathcal I$ is `inside` and $\mathcal F$ is `free`. The frame
resolution is one hop (8 samples in the demo).

### 3. The constrained problem

Moves inside an interval use a penalty with no chirp-rate limit, and
$\lambda_b^{\times}$ in place of $\lambda_b$:

$$
\Pi_t = \begin{cases} P^{\times}_t & t \in \mathcal F \\ P_t & t \notin \mathcal F \end{cases},
\qquad
P^{\times}_t(q\mid q') = \lambda_k\bigl(k-k'-s(b',b)\bigr)^2 + \lambda_b^{\times}(b-b')^2 + \infty\cdot\bigl[\,\lvert k-k'\rvert > D_k\,\bigr].
\tag{10}
$$

$\lambda_b^{\times} = 0$ (the default) leaves the chirp rate free in the
interval; the frequency penalty, the prediction $s$ and the limit $D_k$ are
unchanged. The hinted ridge solves

$$
\max_{q_{0:T-1}}\ \sum_{t} E[t,q_t] - \sum_{t\ge 1}\Pi_t(q_t\mid q_{t-1})
\quad\text{subject to}\quad
\begin{aligned}
&\text{(11a)}\quad \sigma\bigl(k_{t_1^{(j)}}\bigr)\,\sigma\bigl(k_{t_2^{(j)}}\bigr) = -1 \quad \forall j, \\
&\text{(11b)}\quad \sigma(k_t) = \sigma(k_{t-1}) \quad \forall\, t \ge 1,\ t \notin \mathcal F \quad (\texttt{hold\_sign=True}).
\end{aligned}
\tag{11}
$$

Together, (11a) and (11b) mean the ridge changes side exactly once inside each
interval and never elsewhere. With `hold_sign=False`, only (11a) applies: the
ridge may change side any number of times as long as the sides at $t_1$ and
$t_2$ differ. With an empty hint list and `hold_sign=True`, (11b) holds for
every move, so the side never changes.

### 4. Making (11a) local: two copies of the states

(11b) involves only neighbouring frames, so it is a mask on the penalty
matrix. (11a) couples $t_1$ with $t_2$, which can be many frames apart, so
plain Viterbi (eq. 5) cannot apply it. Inside an interval, the state is
therefore extended to $(q, c)$, where copy $c$ records the side the path had
at $t_1$: $\varsigma_c = \sigma(k_{t_1})$, with $\varsigma_0 = +1$ and
$\varsigma_1 = -1$. In practice the DP keeps **two copies** of the score
vector inside each interval: $V^c_t(q)$ for $t \in \mathcal I$, and a single
$V_t(q)$ elsewhere.

```
            outside               ┌───────────── interval (t1, t2) ─────────────┐               outside
frame:  … t1-1 ──hold──► t1 ──enter──► t1+1 ──free──► … ──free──► t2-1 ──leave──► t2 ──hold──► t2+1 …

                         side +1 ────► copy 0 ─────────────────────── copy 0 ────► side -1 at t2
                         side -1 ────► copy 1 ─────────────────────── copy 1 ────► side +1 at t2
```

The recursion (5) becomes one of six cases for each move $t-1 \to t$:

$$
\begin{aligned}
&\textbf{hold}\ (t\notin\mathcal F,\ \texttt{hold\_sign}): &
V_t(q) &= E[t,q] + \max_{q':\ \sigma(k')=\sigma(k)} \bigl[V_{t-1}(q') - P_t(q\mid q')\bigr] \\
&\textbf{plain}\ (t\notin\mathcal F,\ \text{no } \texttt{hold\_sign}): &
V_t(q) &= E[t,q] + \max_{q'} \bigl[V_{t-1}(q') - P_t(q\mid q')\bigr] \\
&\textbf{enter}\ (t-1 = t_1,\ t\in\mathcal I): &
V^c_t(q) &= E[t,q] + \max_{q':\ \sigma(k')=\varsigma_c} \bigl[V_{t-1}(q') - P^{\times}_t(q\mid q')\bigr],\quad c=0,1 \\
&\textbf{inside}\ (t-1\in\mathcal I,\ t\in\mathcal I): &
V^c_t(q) &= E[t,q] + \max_{q'} \bigl[V^c_{t-1}(q') - P^{\times}_t(q\mid q')\bigr],\quad c=0,1 \\
&\textbf{leave}\ (t-1\in\mathcal I,\ t = t_2): &
V_t(q) &= E[t,q] + \max_{q'} \bigl[V^{c(k)}_{t-1}(q') - P^{\times}_t(q\mid q')\bigr],\quad \varsigma_{c(k)} = -\sigma(k) \\
&\textbf{flip}\ (t-1 = t_1,\ t = t_2): &
V_t(q) &= E[t,q] + \max_{q':\ \sigma(k')=-\sigma(k)} \bigl[V_{t-1}(q') - P^{\times}_t(q\mid q')\bigr]
\end{aligned}
\tag{12}
$$

So at $t_2$, a state on side $+1$ can only continue copy 1 (side $-1$ at
$t_1$), and a state on side $-1$ only copy 0. **Flip** is the case of an
interval only one move long.

| move | condition | eq. (12) case | how `_viterbi_flips` implements it |
|---|---|---|---|
| hold | outside, `hold_sign=True` | hold | penalty matrix plus $\infty\cdot[\sigma(k) \ne \sigma(k')]$ |
| plain | outside, `hold_sign=False` | plain | unmasked $P_t$ |
| enter | $t-1 = t_1$, $t < t_2$ | enter | score vector masked: copy $c$ keeps $V_{t_1}(q')$ where $\sigma(k') = \varsigma_c$, else $-\infty$ |
| inside | both frames strictly inside | inside | each copy advances separately with $P^{\times}_t$ |
| leave | $t = t_2$, $t-1 > t_1$ | leave | both copies maximized, then row $k$ takes copy $c(k)$ |
| flip | $t-1 = t_1$, $t = t_2$ | flip | penalty matrix plus $\infty\cdot[\sigma(k) = \sigma(k')]$ |

Back-pointers store $(c', q')$, so backtracking follows the right copy. The
last frame $T-1$ is never strictly inside an interval ($t_2 \le T-1$), so the
backtrack starts at

$$
q^\star_{T-1} = \operatorname*{arg\,max}_q V_{T-1}(q),\qquad
\text{and if } \max_q V_{T-1}(q) = -\infty \text{, no path satisfies (11): } \texttt{ValueError}.
\tag{13}
$$

This happens, for example, when an interval is too short to get from one
side to the other within $D_k$, or with an empty list and `hold_sign` on a
signal that has to cross.

**Why this is exact.** Let $\zeta_t = (q_t, c_t)$, with $c_t$ defined only
for $t \in \mathcal I$. The six cases of eq. (12) allow a move
$\zeta_{t-1} \to \zeta_t$ only if:

* **enter:** $\varsigma_{c_t} = \sigma(k_{t-1})$;
* **inside:** $c_t = c_{t-1}$;
* **leave:** $\varsigma_{c_{t-1}} = -\sigma(k_t)$;
* **hold:** $\sigma(k_t) = \sigma(k_{t-1})$;
* **flip:** $\sigma(k_t) = -\sigma(k_{t-1})$.

Each condition involves only two consecutive states. Within an interval, $c$
is therefore constant and equal to the side at $t_1$, so leaving gives
$\sigma(k_{t_2}) = -\sigma(k_{t_1})$: that is (11a). Conversely, each path
satisfying (11) has exactly one such lift ($c_t$ = the side at $t_1$). $E$ and
$\Pi$ do not depend on $c$, so lifted and original paths have equal cost.
Viterbi is exact for costs made of single-state and consecutive-pair terms,
hence

$$
\max_q V_{T-1}(q) \;=\; \max\bigl\{\, J_\Pi(q_{0:T-1}) \;:\; q_{0:T-1} \text{ satisfies (11a), (11b)} \,\bigr\},
\tag{14}
$$

with $J_\Pi$ the objective in (11). This was also checked against an
exhaustive search over all paths on 300 small random problems: random
penalties, shared end frames, one-move intervals, infeasible cases, with and
without `hold_sign`.

### 5. Coarse and fine passes

**Coarse pass.** There are $K_c = \lceil K/r_k\rceil$ by
$B_c = \lceil B/r_b\rceil$ blocks; block $(i, j)$ holds the cells with
$\lfloor k/r_k\rfloor = i$ and $\lfloor b/r_b\rfloor = j$ (padding cells have
$E = -\infty$). Each block takes the energy of its best cell, and that cell
is its representative $(\hat k_t, \hat b_t)$:

$$
E^{c}[t,i,j] = \max_{(k,b)\in(i,j)} E[t,k,b],\qquad
\bigl(\hat k_t(i,j),\,\hat b_t(i,j)\bigr) = \operatorname*{arg\,max}_{(k,b)\in(i,j)} E[t,k,b].
\tag{15}
$$

Penalties use the block centres $\kappa_i = i\,r_k + (r_k-1)/2$ and
$\mu_j = j\,r_b + (r_b-1)/2$ (in fine-bin units), with limits widened by one
block:

$$
P^{c}\bigl((i,j)\mid(i',j')\bigr) = \lambda_k\bigl(\kappa_i-\kappa_{i'}-s^{c}\bigr)^2 + \lambda_b(\mu_j-\mu_{j'})^2
+ \infty\cdot\bigl[\,\lvert\kappa_i-\kappa_{i'}\rvert > D_k + r_k \ \lor\ \lvert\mu_j-\mu_{j'}\rvert > D_b + r_b\,\bigr],
$$

where $s^{c}$ is $s$ with $\mathrm{sh}$ linearly interpolated at $\mu_{j'}$
and $\mu_j$. The interval version $P^{c\times}$ replaces $\lambda_b$ by
$\lambda_b^{\times}$ and drops the $\mu$ limit, as in eq. (10). The side of a
block is the side of its representative cell, i.e. of the cell the fine pass
will be centred on:

$$
\sigma^{c}_t(i,j) = \sigma\bigl(\hat k_t(i,j)\bigr).
\tag{16}
$$

The coarse pass runs eq. (12) with $E^c$, $P^c$, $P^{c\times}$ and
$\sigma^c_t$. In the two-sided grid, if

$$
(K/2) \bmod r_k = 0 ,
\tag{17}
$$

no block straddles 0 Hz, so $\sigma^c_t(i,j) = \sigma(i\,r_k)$ does not change
with $t$. The masked matrices of the hold and flip cases are then built once
and cached (`static=True`); otherwise they are rebuilt every frame. In the
demo, $K/2 = 256$ and $r_k = 8$, so the cache is used.

**Fine pass.** With $(i_t, j_t)$ the coarse path, the corridor is centred on
$(\bar k_t, \bar b_t) = (\hat k_t(i_t, j_t), \hat b_t(i_t, j_t))$:

$$
\mathcal C_t = \bigl\{(\bar k_t+u,\ \bar b_t+v) : \lvert u\rvert\le r_k,\ \lvert v\rvert\le r_b\bigr\},
\qquad E = -\infty \text{ on cells off the grid}.
\tag{18}
$$

It runs eq. (12) on $\mathcal C_t$ with the exact penalties (3) and (10), and
the exact sides $\sigma(k)$, so the final fine path satisfies (11) exactly.

**Cost.** One move costs $(\text{number of copies}) \times \lvert\text{states at } t\rvert \times \lvert\text{states at } t-1\rvert$:
the work doubles inside intervals and is unchanged elsewhere. Measured: 665 ms
with hints vs 624 ms without (4 s at 1 kHz, demo grid).

## Read-out

The ridge gives bins $(k_t, b_t)$. The returned values are energy-weighted
means of the reassigned operators over a $\pm 2 \times \pm 2$ window around
the ridge (`readout_cells=(2, 2)`); the frequency index wraps on the
two-sided grid:

$$
\hat\omega_t = \frac{\sum_{(k,b)\in W_t} R_\omega[t,k,b]}{\sum_{(k,b)\in W_t} R[t,k,b]},
\qquad
W_t = \bigl\{(k \bmod K,\ b) : \lvert k-k_t\rvert\le 2,\ \lvert b-b_t\rvert\le 2,\ 0\le b<B\bigr\},
\tag{19}
$$

$$
\mathrm{IF}_t = \frac{f_s}{2\pi}\,\hat\omega_t \ \text{(Hz)},\qquad
\mathrm{chirp}_t = \frac{f_s^2}{2\pi}\,\hat\beta_t \ \text{(Hz/s)},
$$

with $\hat\beta_t$ the same mean over $R_\beta$. If the window holds no
energy, the grid values $\omega_{k_t}$, $\beta_{b_t}$ are returned.

## What the hints do not do

**Global sign.** For a real $x$ (and real $g$),

$$
C(t,-\omega,-\beta) = \overline{C(t,\omega,\beta)}
\ \Rightarrow\ (\hat\omega,\hat\beta) \to (-\hat\omega,-\hat\beta)
\ \Rightarrow\ R[t,\bar k,\bar b] = R[t,k,b],\qquad
\bar k = (K-k) \bmod K,\ \ \bar b = B-1-b
\tag{20}
$$

(up to rounding at bin edges). Mirroring also gives
$\mathrm{sh}_{\bar b} = -\mathrm{sh}_b$, so $P_t(\bar q\mid\bar q') = P_t(q\mid q')$,
and $\sigma(\bar k) = -\sigma(k)$ for $k \notin \{0, K/2\}$. The mirrored path
therefore has the same cost and satisfies the same hints: (11a) flips both
ends, (11b) is unchanged. The DP returns one of the two. The demo and the
quick start pick the global sign with an oracle:
$g^\star = \operatorname{arg\,min}_{g\in\{\pm1\}} \sum_m \bigl(g\,\widehat{\mathrm{IF}}(m) - \mathrm{IF}(m)\bigr)^2$
(`if_lib.best_global_sign`).

**Read-out near 0 Hz.** Eq. (11) constrains the bins $k_t$, not
$\hat\omega_t$. If $k_t = K/2$ (0 Hz, $\sigma = +1$), $W_t$ in eq. (19) also
contains $k = K/2-1$ and $K/2-2$, where $\omega_k < 0$, so $\hat\omega_t$ can
be negative. The **ridge** then never changes side outside the intervals, but
the **returned IF** can still cross 0 Hz by a few Hz. In the test set this
happened only in B = 1 Hz, seed 5, where the IF stays near 0 Hz for long
stretches: up to 9 Hz at 0 dB, 5 Hz at 10 dB.

**Wrong hints are obeyed.** A hint where the IF only approaches 0 Hz and turns
back forces a crossing there, by (11a). Envelope dips (`hint source =
detected` in the demo) cannot tell a crossing from a turn-back; B = 1 Hz,
seed 5 shows this at 2.73 s.

**Limits.** Hints need a two-sided grid (`onesided=False`; a one-sided grid
has $\sigma \equiv +1$ and is rejected) and `ridge="dp"`. With
`n_components > 1`, every ridge gets the same hints.

## Grid and limits used by `fcrm_estimate`

`if_lib.fcrm_estimate` derives the chirp-rate grid and jump limits from the
GP prior. The IF is $f_{\text{scale}}$ times a unit-variance GP with a flat
spectrum on $\lvert f\rvert \le B_{\text{GP}}$. The standard deviations of its
slope and curvature then set the grid:

$$
\begin{aligned}
s_{\text{std}} &= f_{\text{scale}}\,\frac{2\pi B_{\text{GP}}}{\sqrt3}\ \ \text{(Hz/s)},
&\qquad c_{\text{std}} &= f_{\text{scale}}\,\frac{(2\pi B_{\text{GP}})^2}{\sqrt5}\ \ \text{(Hz/s}^2\text{)}, \\
\beta_{\max} &= \max(200,\ 4\,s_{\text{std}})\ \ \text{(Hz/s)},
&\qquad \Delta\beta &= \frac{2\beta_{\max}}{n_\beta-1},\qquad \Delta f = \frac{f_s}{K}, \\
D_k &= \Bigl\lceil \frac{6\,s_{\text{std}}\,h}{f_s\,\Delta f} \Bigr\rceil + 4,
&\qquad D_b &= \min\Bigl(\Bigl\lceil \frac{6\,c_{\text{std}}\,h}{f_s\,\Delta\beta} \Bigr\rceil + 2,\ n_\beta-1\Bigr).
\end{aligned}
\tag{21}
$$

Demo defaults: $f_{\text{scale}} = 0.4 f_s/2 = 200$ Hz, $B_{\text{GP}} = 1$ Hz,
$\lambda_k = 0.05$, $\lambda_b = 0.3$.

| quantity | value |
|---|---|
| frames $T$ | 500 (4 s, $f_s$ = 1 kHz, $h$ = 8) |
| frequency bins $K$ | 512, two-sided, $\Delta f$ = 1.953 Hz, 0 Hz at bin 256 |
| chirp-rate bins $B$ ($n_\beta$) | 48 over ±2902 Hz/s, $\Delta\beta$ = 123.5 Hz/s |
| $s_{\text{std}}$ / $c_{\text{std}}$ | 725 Hz/s / 3531 Hz/s² |
| $D_k$ / $D_b$ | 22 / 4 bins |
| coarse states | 64 × 12 = 768 ($r_k$ = 8, $r_b$ = 4) |
| fine corridor | 17 × 9 = 153 cells per frame |

## Parameters

| `estimate_ifs` | `fcrm_estimate` | symbol | default | meaning |
|---|---|---|---|---|
| `zero_crossings` | `zero_crossings` | $[\mathrm{idx}_1^{(j)}, \mathrm{idx}_2^{(j)}]$ | `None` | pair or list of pairs, in samples of `x`; each interval holds exactly one sign change |
| `hold_sign` | `hold_sign` | (11b) | `True` | the intervals list **all** sign changes: no change of side outside them |
| `lam_b_cross` | `lam_c_cross` | $\lambda_b^{\times}$ | `0.0` | chirp-rate penalty per bin² inside the intervals (0 = free); $D_b$ never applies there |

## Results

36 cases: B ∈ {0.5, 1, 2} Hz, SNR ∈ {0, 10} dB, seeds 0–5, fs = 1 kHz, 4 s.
Hints at the true sign changes ± 50 ms; global sign by oracle; 0.3 s at each
end excluded (eq. 24).

| setting | median RMSE | worst RMSE | cases > 20 Hz | time |
|---|---|---|---|---|
| no hints | 3.09 Hz | 295 Hz | 8 / 36 | 624 ms |
| hints, `hold_sign`, chirp rate free (default) | 3.69 Hz | 8.3 Hz | 0 / 36 | 665 ms |
| hints, `hold_sign`, `lam_c_cross = lam_c` | 2.93 Hz | 4.5 Hz | 0 / 36 | 703 ms |
| hints, no `hold_sign` | 3.81 Hz | 296 Hz | 6 / 36 | 685 ms |

Without (11b), most failures remain: they are false crossings outside the
intervals, and the wrong path still satisfies (11a) at the real crossing.
A free chirp rate ($\lambda_b^{\times} = 0$) suits IFs whose slope jumps at the
crossing; on these smooth GP IFs, $\lambda_b^{\times} = \lambda_b$ is more
accurate inside the intervals.

## The demo (`demo_real_signed_hint.py`)

**Hint intervals** (`hint_intervals`). The centres $c_1 < \dots < c_J$ are
either the true sign changes
$\{n : \operatorname{sign} f[n+1] \ne \operatorname{sign} f[n]\}$, each shifted
by $u_j\,\delta$ with $u_j \sim \mathcal U(-1,1)$ and $\delta$ the jitter, or
the envelope dips from `if_lib.turning_points`. With margin $M$, signal length
$N$, midpoints $m_j = (c_j + c_{j+1})/2$, $m_0 = 0$ and $m_J = N-1$:

$$
\mathrm{idx}_1^{(j)} = \operatorname{round}\Bigl(\operatorname{clip}\bigl(\max(c_j - M,\ m_{j-1}),\ 0,\ N-1\bigr)\Bigr),
\qquad
\mathrm{idx}_2^{(j)} = \operatorname{round}\Bigl(\operatorname{clip}\bigl(\min(c_j + M,\ m_j),\ 0,\ N-1\bigr)\Bigr),
\tag{22}
$$

and an interval is dropped if $\mathrm{idx}_2 \le \mathrm{idx}_1$. Neighbouring
intervals meet at most at a midpoint, so eq. (8) holds unless an interval is
shorter than one hop (then eq. 7 widens it to two frames and it may overlap
its neighbour).

**Signed Hilbert with the same intervals**
(`signed_hilbert_estimate` with `loc`). Let $z$ be the analytic signal of $x$
and $\mathrm{env}$ the smoothed envelope of the band-passed signal
(`turning_points_with_locations`). The turning point of interval $j$ is the
envelope minimum $\tau_j$, the sign flips after each $\tau_j$, and the signed
phase differences are low-passed (FIR, cut-off `lp_fc_rel` × B):

$$
\tau_j = \operatorname*{arg\,min}_{\mathrm{idx}_1^{(j)} \le n < \mathrm{idx}_2^{(j)}} \mathrm{env}[n],
\qquad
\mathrm{sgn}[n] = (-1)^{\#\{j\,:\ \tau_j \le n\}},
\qquad
\hat f_H[n] = \mathrm{LP}\Bigl(\mathrm{sgn}[n]\cdot\frac{f_s}{2\pi}\,\angle\bigl(z[n+1]\,\overline{z[n]}\bigr)\Bigr).
\tag{23}
$$

**Scores.** On the grid $m = 0, 4, 8, \dots$ samples within $[0.3\text{ s},\ T_{\text{end}} - 0.3\text{ s}]$,
after the oracle global sign, with $\widehat{\mathrm{IF}}$ linearly
interpolated onto the grid:

$$
e(m) = \widehat{\mathrm{IF}}(m) - \mathrm{IF}(m),\qquad
\text{RMSE} = \sqrt{\operatorname{mean}_m e(m)^2},\qquad
\text{sign correct} = \operatorname{mean}_m\bigl[\operatorname{sign}\widehat{\mathrm{IF}}(m) = \operatorname{sign}\mathrm{IF}(m)\bigr].
\tag{24}
$$

The "trunc" figure is the RMSE over the 90 % smallest $\lvert e\rvert$.

**Panels:**

* two-sided STFT with the true IF, FCRM without hints, FCRM with hints and
  signed Hilbert;
* the FCRM summed over chirp rate;
* the band-passed envelope with the true sign changes and the hint
  intervals;
* the signed error of each method.

Hint intervals are shaded.

| control | CLI flag | effect |
|---|---|---|
| hint source | `--hint-source {true,detected}` | centres in eq. (22): true sign changes, or envelope dips |
| hint margin | `--margin-ms` | $M$ in eq. (22) |
| hint jitter | `--jitter-ms` | $\delta$ in eq. (22) |
| hold sign | `--no-hold` | drop (11b) |
| CR in interval | `--cr-cross {free,lam-c}` | $\lambda_b^{\times} = 0$ or $\lambda_b$ in eq. (10) |

```
python tf_demo/demo_real_signed_hint.py
python tf_demo/demo_real_signed_hint.py --B 2 --seed 2 --margin-ms 80 --jitter-ms 30
python tf_demo/demo_real_signed_hint.py --hint-source detected --save hint.png
```
