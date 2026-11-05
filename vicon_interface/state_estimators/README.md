# State estimation of a Vicon rigid body

This folder holds the estimators. The node uses one of them: `PoseFilter` in `pose_filter.py`, a
Kalman filter of a rigid body's pose. `ViconDataFilter` (`../vicon_data_filter.py`) runs one per
tracked object, and both odometry topics publish its estimate. `KalmanFilter` and
`FloatingBaseLinearStateEstimator` were ported from arc-bridge on `g1_deploy`; the node no longer
uses them.

This page gives the model, where each equation lives in the code, what changed from `g1_deploy`
(5057002), and how the final filter scores. Parameter values are quoted from
`param/vicon_interface.param.yaml`; where the two disagree, the YAML wins.

## Summary

- One constant-velocity Kalman filter per object tracks the rigid body that Vicon reports: position
  and velocity in the world, orientation on $SO(3)$ and angular velocity in the rigid body's frame. Only the
  pose is measured. Both velocities are estimated.
- It steps once per Vicon frame, on the Vicon frame clock, with the exact process noise of that
  step. Host timing reaches only the stamps, and a running minimum removes it there.
- The estimate moves from the rigid body to base_link by the rigid-body formulas. Vicon's
  orientation flicker therefore stays in the rotation, which is smoothed on its own.
- A physical gate drops marker swaps, a Huber weight softens orientation spikes, and every restart
  waits a few frames before publishing.
- `odometry/<obj>` carries the estimate at each frame's capture time, `odometry/filtered/<obj>` the
  same estimate predicted to the publish time.
- With g1 moved by hand, against what `g1_deploy`'s code published: velocity error RMS 47.0 to
  24.6 mm/s, its frame-to-frame jumps 287 to 31 mm/s; angular velocity 188 to 89 mrad/s, jumps 742
  to 100 mrad/s. The price is lag: about 20 ms on velocity, 20 to 28 ms on angular velocity.
- The node ran end to end under real ROS 2 against a replayed SDK. It has not yet run on the robot.

## The problem

Vicon measures a pose and nothing else, at 250 Hz. On `g1_deploy` the published velocities jumped
although the pose looked clean. Two recordings of 2026-11-04, g1 at rest and g1 moved by hand,
showed why:

- Vicon's orientation flickers, by 0.2 to 0.3 mrad at rest and about 1.7 mrad in motion. base_link
  sits below the rigid body (`child_frame_offset_xyz`, see "Frame offsets" in `AGENTS.md`), so every
  flicker also moves base_link. A finite difference over five frames turned that into velocity
  spikes. This was the main cause.
- The finite difference was timed by when the host picked each frame up (2.8 ms late on median, up
  to 9.5 ms), not by the Vicon frame clock. A minor cause.
- The angular velocity came from Euler-angle rates, which is wrong in general (defect 1 below).

## Model

Three frames appear: the world $\mathcal W$, the rigid body $\mathcal M$ that Vicon tracks, and
base_link $\mathcal B$, the published child frame. A left superscript names the frame a vector is
written in; positions and velocities without one are in $\mathcal W$. ${}^{\mathcal W}\mathbf R_{\mathcal M}$
maps $\mathcal M$ coordinates to $\mathcal W$ coordinates, and
${}^{\mathcal M}\mathbf R_{\mathcal W} = ({}^{\mathcal W}\mathbf R_{\mathcal M})^\top$. A tilde marks a
measurement, a hat an estimate, and the superscripts $-$ and $+$ the prior and the posterior at a
step. A bold $\mathbf R$ whose index names no frame, such as $\mathbf R_p$, is a measurement-noise
covariance.

### State

```math
\mathbf p_m,\ \mathbf v_m \in \mathbb R^3,\qquad {}^{\mathcal W}\mathbf R_{\mathcal M} \in SO(3),\qquad
{}^{\mathcal M}\boldsymbol\omega \in \mathbb R^3
```

The filter tracks the rigid body itself, not base_link. The orientation error is taken in $\mathcal M$,
${}^{\mathcal W}\mathbf R_{\mathcal M} = {}^{\mathcal W}\hat{\mathbf R}_{\mathcal M}\operatorname{Exp}(\delta\boldsymbol\theta)$. The state splits into two
blocks with their own covariances: the translation $(\mathbf p_m, \mathbf v_m)$ with
$\mathbf P_{\mathrm{trans}}$ and the rotation $(\delta\boldsymbol\theta, {}^{\mathcal M}\boldsymbol\omega)$ with
$\mathbf P_{\mathrm{rot}}$. Their dynamics and noises do not interact, so the blocks stay uncorrelated
inside the filter. They correlate only once the estimate is moved to base_link.

### Process model

Constant velocity, driven by white acceleration:

```math
\dot{\mathbf p}_m = \mathbf v_m,\qquad \dot{\mathbf v}_m = \mathbf n_a,\qquad
\mathbb E\big[\mathbf n_a(t)\,\mathbf n_a(s)^\top\big] = \sigma_a^2\,\mathbb I_3\,\delta(t-s)
```

```math
{}^{\mathcal W}\dot{\mathbf R}_{\mathcal M} = {}^{\mathcal W}\mathbf R_{\mathcal M}\,[{}^{\mathcal M}\boldsymbol\omega]_\times,\qquad
{}^{\mathcal M}\dot{\boldsymbol\omega} = \mathbf n_\alpha,\qquad
\mathbb E\big[\mathbf n_\alpha(t)\,\mathbf n_\alpha(s)^\top\big] = \sigma_\alpha^2\,\mathbb I_3\,\delta(t-s)
```

$\sigma_a^2$ and $\sigma_\alpha^2$ are `kf_noise_psd`. Over one step $\delta t$ the translation is discretized
exactly:

```math
\begin{bmatrix} \mathbf p_m \\ \mathbf v_m \end{bmatrix}_{k+1} =
\mathbf F_k \begin{bmatrix} \mathbf p_m \\ \mathbf v_m \end{bmatrix}_{k} + \mathbf w_k,\qquad
\mathbf F_k = \begin{bmatrix} \mathbb I_3 & \delta t\,\mathbb I_3 \\ \mathbf 0 & \mathbb I_3 \end{bmatrix},\qquad
\mathbf Q_{\mathrm{trans},k} = \sigma_a^2 \begin{bmatrix} \tfrac{1}{3}\delta t^3\,\mathbb I_3 & \tfrac{1}{2}\delta t^2\,\mathbb I_3 \\
\tfrac{1}{2}\delta t^2\,\mathbb I_3 & \delta t\,\mathbb I_3 \end{bmatrix}
```

$\mathbf Q_{\mathrm{trans},k}$ integrates the noise exactly, so two steps of $\delta t/2$ equal one step of
$\delta t$: a dropped frame is just a longer step.

The rotation moves its mean along the exponential map and its error to first order:

```math
{}^{\mathcal W}\hat{\mathbf R}_{\mathcal M_{k+1}}^- = {}^{\mathcal W}\hat{\mathbf R}_{\mathcal M_k}^+\operatorname{Exp}(\boldsymbol\phi),\qquad
{}^{\mathcal M}\hat{\boldsymbol\omega}_{k+1}^- = {}^{\mathcal M}\hat{\boldsymbol\omega}_k^+,\qquad
\boldsymbol\phi = {}^{\mathcal M}\hat{\boldsymbol\omega}_k^+\,\delta t
```

```math
\begin{bmatrix} \delta\boldsymbol\theta \\ \delta\boldsymbol\omega \end{bmatrix}_{k+1} =
\begin{bmatrix} \operatorname{Exp}(\boldsymbol\phi)^\top & \delta t\,\mathbf J_r(\boldsymbol\phi) \\ \mathbf 0 & \mathbb I_3 \end{bmatrix}
\begin{bmatrix} \delta\boldsymbol\theta \\ \delta\boldsymbol\omega \end{bmatrix}_{k} + \mathbf w_k
```

$\mathbf J_r$ is the right Jacobian of $SO(3)$. The rotation's $\mathbf Q_{\mathrm{rot},k}$ is
$\mathbf Q_{\mathrm{trans},k}$ with $\sigma_\alpha^2$ in place of $\sigma_a^2$; it ignores how far $\mathcal M$ turns
within one 4 ms frame.

$\delta t$ always comes from the Vicon frame clock, $\delta t = (n_k - n_{k-1})/f$, with the frame
number $n$ and the frame rate $f$ from the SDK. Host time never enters it.

### Measurement model

Each frame measures the rigid body's pose:

```math
\tilde{\mathbf p}_m = \mathbf p_m + \mathbf n_p,\quad \mathbf n_p \sim \mathcal N(\mathbf 0,\ \mathbf R_p),\qquad
{}^{\mathcal W}\tilde{\mathbf R}_{\mathcal M} = {}^{\mathcal W}\mathbf R_{\mathcal M}\operatorname{Exp}(\mathbf n_\theta),\quad
\mathbf n_\theta \sim \mathcal N(\mathbf 0,\ \mathbf R_\theta)
```

$\mathbf R_p = \sigma_p^2\,\mathbb I_3$ and $\mathbf R_\theta = \sigma_\theta^2\,\mathbb I_3$, with $\sigma_p$ and $\sigma_\theta$ from
`kf_measurement_std`. Against the prediction $\hat{\mathbf p}_m^-$, ${}^{\mathcal W}\hat{\mathbf R}_{\mathcal M}^-$ the
innovations are

```math
\boldsymbol\nu_p = \tilde{\mathbf p}_m - \hat{\mathbf p}_m^-,\qquad
\boldsymbol\nu_\theta = \operatorname{Log}\big(({}^{\mathcal W}\hat{\mathbf R}_{\mathcal M}^-)^\top\,{}^{\mathcal W}\tilde{\mathbf R}_{\mathcal M}\big) \approx \delta\boldsymbol\theta + \mathbf n_\theta
```

so each block observes its first three states, $\mathbf H = [\,\mathbb I_3\ \ \mathbf 0\,]$. Velocity and angular velocity are
never measured; the model infers them from successive poses.

### Update and the rules around it

Each block takes the Kalman update in Joseph form, with $\boldsymbol\nu$ and $\mathbf R_\nu$ its innovation
and noise covariance ($\boldsymbol\nu_p$ and $\mathbf R_p$, or $\boldsymbol\nu_\theta$ and $\mathbf R_\theta$):

```math
\mathbf S_k = \mathbf H\mathbf P_k^-\mathbf H^\top + \mathbf R_\nu,\qquad
\mathbf K_k = \mathbf P_k^-\mathbf H^\top\mathbf S_k^{-1},\qquad
\delta\hat{\mathbf x}_k = \mathbf K_k\boldsymbol\nu,\qquad
\mathbf P_k^+ = (\mathbb I - \mathbf K_k\mathbf H)\,\mathbf P_k^-\,(\mathbb I - \mathbf K_k\mathbf H)^\top + \mathbf K_k\mathbf R_\nu\mathbf K_k^\top
```

The translation adds $\delta\hat{\mathbf x}_k$. The rotation injects it:
${}^{\mathcal W}\hat{\mathbf R}_{\mathcal M}^+ = {}^{\mathcal W}\hat{\mathbf R}_{\mathcal M}^-\operatorname{Exp}(\delta\hat{\boldsymbol\theta}_k)$ and
${}^{\mathcal M}\hat{\boldsymbol\omega}^+ = {}^{\mathcal M}\hat{\boldsymbol\omega}^- + \delta\hat{\boldsymbol\omega}_k$.

- **Gate.** A frame with $\lVert\boldsymbol\nu_p\rVert > g_p$ or $\lVert\boldsymbol\nu_\theta\rVert > g_\theta$ (`kf_gate`,
  in metres and radians) is a marker swap or a flipped fit. The filter keeps its prediction and
  skips the frame. After `kf_max_rejections` in a row it restarts at the measurement, since the rigid
  body really moved. Motion stays far below both thresholds.
- **Huber weight on the rotation.** With the Mahalanobis distance
  $d = \sqrt{\boldsymbol\nu_\theta^\top\mathbf S_k^{-1}\boldsymbol\nu_\theta}$, an innovation with $d > \kappa$ (`kf_rot_huber_k`) is
  fused with $\mathbf R_\theta$ scaled by $d/\kappa$. An orientation spike then pulls the estimate less. Below
  $\kappa$ the update is the plain one.
- **Restart.** The filter starts at the measured pose, at rest ($\mathbf v_m = {}^{\mathcal M}\boldsymbol\omega = \mathbf 0$),
  with its covariance from $\sigma_p$, $\sigma_\theta$ and `kf_initial_std` ($\sigma_{v,0}$, $\sigma_{\omega,0}$). This happens at start, after an
  offline gap, after a frame step it cannot fuse (backwards, as when Vicon restarts its count, or
  longer than `vicon_offline_timeout`) and at the gate's limit. Each time, it publishes again only
  after `rolling_window_size` frames.

### From the rigid body to base_link

base_link is fixed on the rigid body at the offset ${}^{\mathcal M}\mathbf r_{m,b}$ with the orientation
${}^{\mathcal M}\mathbf R_{\mathcal B}$ (`child_frame_offset_xyz`, `child_frame_offset_rpy_deg`). The estimate moves there by the
rigid-body formulas

```math
\mathbf p_b = \mathbf p_m + {}^{\mathcal W}\mathbf R_{\mathcal M}\,{}^{\mathcal M}\mathbf r_{m,b},\qquad
\mathbf v_b = \mathbf v_m + {}^{\mathcal W}\mathbf R_{\mathcal M}\,\big({}^{\mathcal M}\boldsymbol\omega \times {}^{\mathcal M}\mathbf r_{m,b}\big),\qquad
{}^{\mathcal W}\mathbf R_{\mathcal B} = {}^{\mathcal W}\mathbf R_{\mathcal M}\,{}^{\mathcal M}\mathbf R_{\mathcal B},\qquad
{}^{\mathcal B}\boldsymbol\omega = {}^{\mathcal B}\mathbf R_{\mathcal M}\,{}^{\mathcal M}\boldsymbol\omega
```

and the covariance by their Jacobian $\mathbf D$, $\mathbf P_b = \mathbf D\,\mathbf P\,\mathbf D^\top$ with
$\mathbf P = \operatorname{blockDiag}(\mathbf P_{\mathrm{trans}}, \mathbf P_{\mathrm{rot}})$:

```math
\delta\mathbf p_b = \delta\mathbf p_m - {}^{\mathcal W}\mathbf R_{\mathcal M}[{}^{\mathcal M}\mathbf r_{m,b}]_\times\,\delta\boldsymbol\theta,\qquad
\delta\mathbf v_b = \delta\mathbf v_m - {}^{\mathcal W}\mathbf R_{\mathcal M}[{}^{\mathcal M}\boldsymbol\omega\times{}^{\mathcal M}\mathbf r_{m,b}]_\times\,\delta\boldsymbol\theta
- {}^{\mathcal W}\mathbf R_{\mathcal M}[{}^{\mathcal M}\mathbf r_{m,b}]_\times\,\delta\boldsymbol\omega,\qquad
\delta\boldsymbol\theta_b = {}^{\mathcal B}\mathbf R_{\mathcal M}\,\delta\boldsymbol\theta,\qquad
\delta\boldsymbol\omega_b = {}^{\mathcal B}\mathbf R_{\mathcal M}\,\delta\boldsymbol\omega
```

This is why the filter tracks the rigid body. Fed base_link's pose, it would take the orientation flicker
times the lever arm as position noise and smooth both with one setting. Tracking the rigid body keeps the
flicker in the rotation block, whose density is set on its own.

The messages carry $\mathbf p_b$, ${}^{\mathcal W}\mathbf R_{\mathcal B}$, the velocity in $\mathcal B$,
${}^{\mathcal B}\mathbf v_b = {}^{\mathcal B}\mathbf R_{\mathcal W}\,\mathbf v_b$, and ${}^{\mathcal B}\boldsymbol\omega$. The
covariances follow nav_msgs/Odometry: the pose in the parent frame, its rotation part turned to the
world axes (${}^{\mathcal W}\mathbf R_{\mathcal B}\,\mathbf P_{b,\theta\theta}\,{}^{\mathcal B}\mathbf R_{\mathcal W}$); the twist in the child frame
(${}^{\mathcal B}\mathbf R_{\mathcal W}\,\mathbf P_{b,vv}\,{}^{\mathcal W}\mathbf R_{\mathcal B}$ and $\mathbf P_{b,\omega\omega}$); cross terms included.

### What the densities mean

At 250 Hz the steady filter acts, per axis, as a second-order Butterworth low-pass on the true rate:

```math
\omega_n = \left(\frac{\sigma_a^2}{\sigma_p^2\,\delta t}\right)^{1/4}\ \text{(translation)},\qquad
\omega_n = \left(\frac{\sigma_\alpha^2}{\sigma_\theta^2\,\delta t}\right)^{1/4}\ \text{(rotation)},\qquad
\text{lag of the rate estimate} \approx \frac{\sqrt{2}}{\omega_n}
```

with $\delta t$ the frame period. The chosen values give about 14 Hz and 16 ms for the translation, and
about 10 Hz and 23 ms for the rotation. Scaling a density and its $\sigma^2$ together ($\sigma_a^2$ with $\sigma_p^2$,
$\sigma_\alpha^2$ with $\sigma_\theta^2$) leaves the estimate as it is; it only rescales the published covariances and the Huber distance.

### Timing and the two odometry topics

- The filter's time $t_k$ is the Vicon time since its last restart: frame numbers over the frame
  rate.
- The capture time on the host clock is $s_k = t_k + \min_j\,(h_j - \ell_j - t_j)$, the minimum over
  the frames of the last `kf_clock_window`. $h_j$ is when the host handled frame $j$ and $\ell_j$ the
  latency the SDK reported for it. Host delays only add to $h_j - \ell_j - t_j$, so the minimum
  strips them.
- `odometry/<obj>` publishes the estimate at $t_k$, moved to base_link and stamped $s_k$, one message
  per frame. A consumer that compensates latency itself reads this one.
- `odometry/filtered/<obj>` is a timer at `filtered_odom_dt`. At host time $h$ it predicts the latest
  estimate ahead by $\min(\max(h - s_k, 0), \Delta_{\max})$, with $\Delta_{\max}$ = `kf_max_delay`,
  moves it to base_link and stamps it now. The filter itself does not move. The topic pauses after
  `vicon_offline_timeout` without a frame, while frames come without a frame rate, and during a
  warm-up.
- `vicon_pose/<obj>` and TF keep Vicon's own pose of base_link, unfiltered.

## How it is written

| Piece | Code |
|---|---|
| $[\cdot]_\times$, $\operatorname{Exp}$, $\operatorname{Log}$ and $\mathbf J_r$ on $SO(3)$ | `pose_filter.py`: `skew`, `so3_exp`, `so3_log` (with its branch near $\pi$), `so3_right_jacobian` |
| $\mathbf F$ and $\mathbf Q$ of one axis | `kinematic_blocks(dt)`; `np.kron` with $\mathbb I_3$ builds the 6×6 blocks |
| prediction of mean and covariance | `PoseFilter._propagate` |
| Joseph update, Mahalanobis distance | `_correct`, `_mahalanobis` |
| gate, Huber weight, outcome | `PoseFilter.update`, which returns `'accepted'`, `'rejected'`, `'reset'` or `'stale'` |
| restart at rest | `PoseFilter.reset` |
| prediction that leaves the filter as it is | `PoseFilter.predicted` |
| move to base_link, Odometry covariances | `PoseEstimate.transported`, `PoseEstimate.odometry_covariances` |
| frame clock, restarts, warm-up, clock offset, watchdog | `../vicon_data_filter.py`: `ViconDataFilter.add_pose`, `update`, `get_frame_state`, `get_filtered_state` |
| mm to m, Euler angles, rigid-body and base_link poses | `../vicon_data_buffer.py`: `ViconDataBuffer.add_pose` |
| parameters, messages, clipping | `../vicon_stream_node.py` |

`test/test_pose_filter.py` checks the model against closed forms: exact tracking of
constant-velocity motion, $\mathbf Q$ against Van Loan's matrix exponential, the noise of the velocity at
rest against the steady Riccati solution, dropped frames, the gate and the Huber weight, a 90°
heading, yaw through $\pi$, and the transport against analytic rigid motion and against sampled
covariances. `test/test_vicon_data_filter.py` checks the timing and the topics' rules with a fake
clock.

## Compared with g1_deploy

### What g1_deploy computed

On `g1_deploy` (5057002) the twist came from a finite difference, and a Kalman filter smoothed it
for `odometry/filtered`.

The **finite difference** (`ViconDataBuffer`) ran over the last $N$ = `rolling_window_size` = 5
poses of base_link, timed by the host stamps $\tau = h - \ell$:

```math
{}^{\mathcal B}\mathbf v_b^{\mathrm{fd}} = {}^{\mathcal B}\tilde{\mathbf R}_{\mathcal W}\,\frac{\tilde{\mathbf p}_{b,N} - \tilde{\mathbf p}_{b,1}}{\tau_N - \tau_1},\qquad
{}^{\mathcal B}\boldsymbol\omega^{\mathrm{fd}} = {}^{\mathcal B}\tilde{\mathbf R}_{\mathcal W}\,\frac{\operatorname{wrap}(\tilde{\boldsymbol\Theta}_N - \tilde{\boldsymbol\Theta}_1)}{\tau_N - \tau_1}
```

$\tilde{\boldsymbol\Theta}$ are the measured sxyz Euler angles and ${}^{\mathcal B}\tilde{\mathbf R}_{\mathcal W}$ the transpose
of the latest measured attitude. `odometry/<obj>` published Vicon's pose with this twist, stamped
$h - \ell$, with zero covariances.

The **Kalman filter** (`ViconLinearStateEstimator` inside `ViconDataFilter`) had the state
$\mathbf x = (\mathbf p_b, \mathbf v_b)$ of base_link in $\mathcal W$ and measured both:

```math
\mathbf x_{k+1} = \mathbf F(\delta t_{\mathrm{tick}})\,\mathbf x_k + \mathbf w_k,\quad \mathbf w_k \sim \mathcal N(\mathbf 0, \mathbf Q),\qquad
\mathbf y_k = \begin{bmatrix} \tilde{\mathbf p}_b + {}^{\mathcal W}\tilde{\mathbf R}_{\mathcal B}\,{}^{\mathcal B}\mathbf v_b^{\mathrm{fd}}\,\Delta \\
{}^{\mathcal W}\tilde{\mathbf R}_{\mathcal B}\,{}^{\mathcal B}\mathbf v_b^{\mathrm{fd}} \end{bmatrix} = \mathbf x_k + \mathbf n_k,\quad
\mathbf n_k \sim \mathcal N(\mathbf 0, \mathbf R_{\mathbf y})
```

- It predicted on every timer tick with a fixed $\delta t_{\mathrm{tick}}$ = `filtered_odom_dt` = 4 ms and a
  constant $\mathbf Q$ per tick,
  $\operatorname{diag}(10^{-8}\,\mathrm{m^2}\,\mathbb I_3,\ 1.44\times10^{-4}\,\mathrm{(m/s)^2}\,\mathbb I_3)$, however long the tick
  really took.
- It corrected on every frame with
  $\mathbf R_{\mathbf y} = \operatorname{diag}(4\times10^{-8}\,\mathrm{m^2}\,\mathbb I_3,\ 2.5\times10^{-4}\,\mathrm{(m/s)^2}\,\mathbb I_3)$. The
  position was moved to now by the frame age $\Delta = \min(h - \tau, 20\ \mathrm{ms})$; the
  velocity measurement was the finite difference.
- `odometry/filtered` published $\hat{\mathbf p}_b$ and $({}^{\mathcal W}\tilde{\mathbf R}_{\mathcal B_{\mathrm{now}}})^\top\hat{\mathbf v}_b$, Vicon's
  Euler angles pushed forward by the frame age times their finite-difference rate, and
  ${}^{\mathcal B}\boldsymbol\omega^{\mathrm{fd}}$ unchanged, stamped now.

### What changed

| | `g1_deploy` | now |
|---|---|---|
| frame filtered | base_link | the rigid body, moved to base_link at the output |
| state | $\mathbf p_b$, $\mathbf v_b$ | $\mathbf p_m$, $\mathbf v_m$; ${}^{\mathcal W}\mathbf R_{\mathcal M}$ on $SO(3)$, ${}^{\mathcal M}\boldsymbol\omega$ |
| measurement | position and finite-difference velocity | pose only: position and orientation |
| orientation | not filtered | filtered on $SO(3)$, with its own density |
| angular velocity | ${}^{\mathcal B}\tilde{\mathbf R}_{\mathcal W}$ times Euler-angle rates | ${}^{\mathcal M}\boldsymbol\omega$, a state |
| time step | timer tick, fixed 4 ms, $\mathbf Q$ per tick | Vicon frame clock, $\mathbf Q(\delta t)$ exact |
| `odometry/<obj>` | Vicon's pose, finite-difference twist, no covariances | the filter at the capture time, with covariances |
| `odometry/<obj>` stamp | host receive time minus the latency | frame clock plus the minimum host offset |
| `odometry/filtered/<obj>` | filter $\hat{\mathbf p}_b$, $\hat{\mathbf v}_b$; Euler angles pushed forward without a cap; ${}^{\mathcal B}\boldsymbol\omega^{\mathrm{fd}}$ | the filter predicted to now, capped at `kf_max_delay` |
| outliers | none | physical gate with restart; Huber weight on the rotation |
| restarts | only after an offline gap | also after a step it cannot fuse or a lasting rejection, each with a warm-up |
| a frame that fails to parse | the previous frame published again | skipped |
| `poll_vicon_dt` | 4 ms | 2 ms |

### Defects found, and what became of them

1. **Angular velocity from Euler-angle rates.** ${}^{\mathcal B}\tilde{\mathbf R}_{\mathcal W}\,\dot{\tilde{\boldsymbol\Theta}}$ is the body
   rate ${}^{\mathcal B}\boldsymbol\omega$ only for a pure yaw or near zero heading and pitch. Rolling at 1 rad/s
   about the x axis of $\mathcal M$, `g1_deploy`'s buffer read +1 rad/s where the truth was -1 (g1's offset turned 180° in yaw, heading 0), and
   (0, -0.94, 0.33) instead of (1, 0, 0) at a 90° heading without an offset. Now ${}^{\mathcal M}\boldsymbol\omega$ is a
   state, tested at a 90° heading and through yaw ±π.
2. **Finite difference timed by the host.** Every host delay went straight into $\tau_N - \tau_1$,
   16 ms across the window. On the moving recording, timing the same difference by frame number
   lowered its velocity RMS only from 46.3 to 43.8 mm/s; the flicker mattered more. Now no host time
   enters the filter.
3. **The finite difference as a measurement.** Its errors are correlated from frame to frame,
   because the windows overlap, and they carry defects 1 and 2 and the flicker; the filter could
   only lag them. $\mathbf Q$ did not follow the time between predictions, and predictions followed the
   timer, not the frames. Now only poses are measured, on the frame clock, with $\mathbf Q(\delta t)$. The
   619a73e draft had already moved the filter to the frame clock with a position-only measurement.
4. **base_link filtered directly.** The flicker times the lever arm entered the position. Now the
   rigid body is filtered and the estimate transported.
5. **A frame that failed to parse went out twice.** The `continue` after the parse error was
   commented out, so the previous frame was published again under a new stamp on `odometry/<obj>`,
   `vicon_pose/<obj>` and TF. Reproduced under ROS. Now the frame is skipped.
6. **Uncapped rotation prediction.** `odometry/filtered` pushed the Euler angles forward by the whole
   frame age, up to `vicon_offline_timeout`, at the finite-difference rate. The draft then capped the
   position but not the angles: in a 60 ms gap the yaw ran 60 ms ahead and the position 20 ms. Now
   both come from one prediction, capped at `kf_max_delay`.
7. **`poll_vicon_dt` equal to the frame period.** The sleep that paces the loop overshoots a little,
   so the loop fell behind frame after frame until it dropped one: every 0.26 s, 1.4 % of the
   frames, with a 0 to 4 ms sawtooth in the pick-up delay. At 2 ms the simulation dropped no frame
   in motion and one at rest.
8. **No covariances** on the odometry topics. Now the filter's, moved to base_link.

Two more were in the 619a73e draft only, and are fixed as well. Without a frame rate,
`odometry/filtered` kept publishing a frozen state. A restart while frames kept coming published the
at-rest state at once, a frame of zero velocity on both topics.

### Parameters

| `g1_deploy` | now |
|---|---|
| `kf_process_noise`: 6 variances per predict step | `kf_noise_psd`: $\sigma_a^2$, $\sigma_\alpha^2$ |
| `kf_measurement_noise`: 6 variances, position and finite-difference velocity | `kf_measurement_std`: $\sigma_p$, $\sigma_\theta$ |
| `kf_initial_covariance`: 6 variances | `kf_initial_std`: $\sigma_{v,0}$, $\sigma_{\omega,0}$ for velocity and angular velocity after a restart |
| none | `kf_gate`, `kf_max_rejections`, `kf_rot_huber_k`, `kf_clock_window` |
| `kf_max_delay`: cap on moving a measurement to now | same name: cap on the prediction to the publish time |
| `rolling_window_size`: the finite-difference window | same name: the warm-up after a restart |
| `poll_vicon_dt`: 0.004 | 0.002 |

The values when this page was written:

| Parameter | Value |
|---|---|
| `kf_noise_psd` | 0.01 m²/s³ (translation), 0.05625 rad²/s³ (rotation) |
| `kf_measurement_std` | 0.2 mm, 1 mrad |
| `kf_initial_std` | 1 m/s, 2 rad/s |
| `kf_gate` | 0.02 m, 0.2 rad |
| `kf_max_rejections` | 5 |
| `kf_rot_huber_k` | 2 |
| `kf_max_delay` | 20 ms |
| `kf_clock_window` | 2 s |
| `vicon_offline_timeout` | 0.1 s |
| `rolling_window_size` | 5 frames |
| `poll_vicon_dt`, `filtered_odom_dt` | 2 ms, 4 ms |

The densities and the Huber constant came from sweeps on both recordings. A coarse one crossed three
translation densities, three rotation densities and three Huber constants, plus six
constant-acceleration settings; a finer one tried 27 settings around the choice. The rule was the
lowest error RMS of the four outputs over both recordings, among the settings whose velocity and
angular-velocity jumps stayed small. The optimum is flat: the best neighbour lowered the mean RMS by
1.6 % and raised the jumps by 3 %. The best constant-acceleration setting had 23 % more velocity RMS
and 26 % more angular-velocity RMS in motion than the chosen filter, so that model was removed. The
finite difference went too, once the filter beat it.

## Results

Each output is compared with a zero-phase reference: a quadratic fit over ±12 frames of the rigid
body's position and orientation increments, moved to base_link. RMS is the error's RMS. A jump is the
99.9th percentile of the error's change from one frame to the next; the motion itself cancels, and
what remains are the steps a consumer sees. The recording at rest leaves out 18 to 25 s, where g1
was touched.

`odometry/<obj>`, RMS / jump:

| Output | in motion, `g1_deploy` path | in motion, now | at rest, `g1_deploy` path | at rest, now |
|---|---|---|---|---|
| position (mm) | 0.42 / 4.54 | 0.41 / 0.69 | 0.05 / 0.43 | 0.03 / 0.12 |
| rotation (mrad) | 1.49 / 15.1 | 1.57 / 2.50 | 0.18 / 1.53 | 0.10 / 0.40 |
| velocity (mm/s) | 47.0 / 287 | 24.6 / 31.1 | 3.32 / 32.6 | 1.81 / 7.04 |
| angular velocity (mrad/s) | 188 / 742 | 89.0 / 99.8 | 12.0 / 112 | 5.83 / 21.4 |

The `g1_deploy` path is what the node published on `odometry/g1` during the recordings: Vicon's pose
and the host-timed finite difference. That code is the same on `g1_deploy` and on the 619a73e draft
the recordings ran. "Now" is the final filter replayed on the same frames. `g1_deploy`'s own
`odometry/filtered` was not recorded, so it has no column.

- Jumps fall 3.6 to 9 times on every output. Velocity and angular-velocity RMS fall by half.
- Rotation RMS in motion rises from 1.49 to 1.57 mrad: smoothing the flicker costs some lag. Its
  jumps fall six times.
- The lag, the shift that minimizes the RMS, is about 20 ms on velocity and 20 ms (at rest) to 28 ms
  (in motion) on angular velocity, close to the 16 and 23 ms the bandwidths predict.

`odometry/filtered/<obj>` extrapolates up to 20 ms with the constant-velocity model, so it is a little
noisier. RMS / jump in motion: 0.69 / 1.06 mm, 2.79 / 5.05 mrad, 30.8 / 37.3 mm/s,
106 / 119 mrad/s. At rest: 0.04 / 0.21 mm, 0.14 / 0.64 mrad, 1.95 / 8.57 mm/s, 6.50 / 23.6 mrad/s.

The whole node was also run as on the robot. ROS 2 Jazzy ran in Docker with the real rclpy,
tf_transformations and rosbag2. Only the Vicon SDK was replaced, by a stand-in that replayed the
recorded frames at their recorded arrival times, for g1 and a static box. Faults were injected: an
unparsable frame, gaps of 60 and 300 ms, 200 ms without a frame rate, g1 missing for 50 ms and a
5 cm marker jump. Every check passed:

- `odometry/<obj>` equalled the unit-tested filter bit for bit on every frame.
- All values were finite, the quaternions unit, the covariances symmetric and positive
  semi-definite, the stamps increasing.
- The marker jump moved `vicon_pose` by 50 mm and `odometry/g1` by 0.48 mm. Each fault paused or
  skipped what it should; the node logged the parse error, the missing object, the missing frame
  rate and the offline gap.
- `odometry/<obj>` ran at 245 to 250 Hz and `odometry/filtered` at 248 to 250 Hz. The node used 36 to
  39 % of one core for two objects. On arrival `odometry/<obj>` was 5.1 ms old on median, the Vicon
  latency included.

## Limits

- A causal filter trades noise for lag. The lags above are the price of the smoothing, and the error
  RMS already counts them.
- The flicker is smoothed, not removed.
- The densities fit g1 moved by hand. Walking has sharper accelerations, which a constant-velocity
  model shows as lag. If the velocity looks late on the robot, raise `kf_noise_psd`; if it looks
  noisy, lower it. The ratios $\sigma_a^2/\sigma_p^2$ and $\sigma_\alpha^2/\sigma_\theta^2$ are what count.
- Not verified yet: the real SDK's frame delivery, the lab machine's load and the robot's own motion.
- The recordings and the script behind these numbers were removed from the tree. They remain in the
  history at f404fb6 (`data/2026-11-04_g1_static_dynamic/`, `replay_vicon_log.py`).
