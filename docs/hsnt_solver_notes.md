# Design notes and measurements behind `mbirtorch.hsnt`

This document holds the experiment-derived reasoning that used to live in code comments and docstrings:
what was measured, on which data, and what decision it drove. The code keeps short pointers here. All
measurements are on the three-material phantom of `generate_hyper_data` (Ni, Cu, Al; K = 1200 bins) unless
stated; "dose" is `dosage_rate`, the open-beam counts per pixel and bin; "SNR" of spectra is the direct-fit
spectral SNR in dB (see `docs/hsnt_reading_list.md` for the methods themselves). Dates are 2026-09-02 to 09-04.

## 1. Loss and numerics

**Stable NNAL.** The non-negative attenuation loss `L(X) = sum exp(-X) + T X` is evaluated in the shifted form
`sum T * phi(X + log T)`, `phi(u) = exp(-u) - 1 + u`, with a Taylor branch below `(24 eps)^(1/4)` so that the
loss goes to exactly zero for a perfect fit and the shifted Poisson negative log-likelihood is recovered.
Zero counts (T marked as 1e-30 upstream) fall back to `exp(-X)`.

**Float64 accumulation.** A per-bin sum over P pixels in float32 has an ulp of 0.06 near 1e6 and 8 near 1e8; an
H step that improves a bin by less than that is invisible to a line search. All row-wise sums and the
streaming accumulators are float64. On the 10M-pixel streaming run this did not lift the plateau by itself
(see section 6): the plateau was estimator bias, not precision, and float64 polish reproduced float32 to every
printed digit.

**Armijo noise floor (`_ARMIJO_FLOOR = 4`).** In float32 the noise in a line-search comparison is not the
sum's roundoff (a zero-mean random walk that float64 accumulation removes) but per-element truncation:
elements whose step `alpha*B` is below `ulp(X)` do not move, so the measured decrease is a biased
truncation, linear in the row length. A floor `c * eps32 * |row loss|` is the right form; `/sqrt(n)` brought
the spurious-backtracking pathology back (757 backtracks per 60 block steps). With float64 sums the constant
could drop from 8 to 4: backtracks per 60 block steps at c = 0.5/1/2/4/8 were 650/468/160/27/15 at P = 4096 and
543/473/236/48/15 at P = 16384, the converged loss identical to the last digit in every case. c <= 2 is
pathological. The floor sets wasted evaluations at small P and the radius of the noise ball H may wander in at
large P.

**Trust-region floor (`_TRUST_FLOOR = 1e-3` of the mean row scale).** The per-row trust region `16 *
max(|row|, floor)` had `eps` as its floor. A row pinned near zero (values of 1e-112 occur in float64) then grows
by at most 16x per step, and the loss-based stopping rule fires during that crawl: a float64 W solve stopped
with a projected gradient 3e5 times the float32 one. In float32, eps (1.2e-7 against rows of 0.08) happened to
be a workable floor.

**Epsilon-active set (`_ACTIVE_TOL = 1e-6` of the mean row scale).** A component that hits the feasibility
limit lands at `V - (V/d) d`, a residue of ~1e-17 V in float64 that is formally free; the row's shared step
length `min(V/d)` is then ~0 and the row freezes at a non-stationary point. float32 rounds the residue to
exactly zero, which is why it "worked". Components within `_ACTIVE_TOL` of zero with an outward gradient are
treated as at the bound and snapped to zero (Bertsekas). Afterwards both precisions converge monotonically to
the same loss, lower than the old float32 result (8.9796973e5 vs 8.979758e5 on the 4k test), float64 reaching a
projected gradient of 6e-16.

**Compiling the kernels (`_kernels(compile_mode)`).** `torch.compile` of the elementwise kernels and the block
step measured 2.2x on a whole joint_newton solve at P = 4096, 2.9x at P = 16384, 3.8x on a block_newton solve;
`max-autotune` gave no steady-state gain for 2.4x the compile time. Compiling costs 4.5 s with a warm inductor
cache, ~18 s cold, so it is opt-in.

## 2. Initialization

**NNDSVDa fill.** Zero entries of the NNDSVD factors are filled at factor scale, `c * sqrt(mean)` with c = 1.
The ladder that led there (loss at 4k, dose 3): 904,026 at mean/100 (spurious basin), 902,547 at c = 0.1,
900,323 at c = 0.3, 898,850 at the classic mean fill, 898,514 at c = 1. With the mean fill on the attenuation
matrix (mean 34.5) every noise component received 1190. The initialization is computed on `-log T` with zero
counts floored at half a count relative to the smallest genuine transmission; flooring at 1e-30 instead gave
`X max 2445`.

## 3. Solvers

**joint_newton (recommended).** Five block warm-up steps, then a matrix-free truncated Newton on (W, H): the
Hessian-vector product costs about five GEMMs, block-diagonal preconditioned CG with `cg_max = 10`,
Eisenstat-Walker forcing, Levenberg damping. `warmup_steps = 5, cg_max = 10` were chosen by interleaved A/B at
equal converged quality against 10/20: 0.91x at dose 3 on 64x64 (a real 9% loss), 2.43x at dose 100, 1.31x at
dose 3 on 128x128. Reaches machine precision on noiseless data (loss ~1e-24 to 1e-27, |X - X_true| ~5e-16).
Within 0.002% of its optimum at `rel_tol = 1e-6`; ~1.2 s at P = 4096 in float32 on a laptop GPU; ~4.5 us/px on
an H100.

**block_newton.** Exact projected Newton per pixel / per bin with the Khatri-Rao Hessian GEMM, batched
Cholesky, two-metric projection, KKT release of bound entries with an inward gradient, trust region,
elementwise line search. Linear convergence: plateaus around 1e-7 where an exact factorization exists; at dose
100 stopped 1.4% short at 1e-6 and 0.001% short at 1e-8. It remains the warm-up and the fixed-H W solver.

**multiplicative.** Damped multiplicative update with the shifted step `V <- max((V + d) r - d, 0)`, whose
fixed-point set equals the KKT set for any d > 0 (so zeros can be resurrected: on the demo 5.5% of W started
at zero and never moved under the plain update; 1.77% of entries were zero with a negative gradient).
Random re-seeding of components dead in both factors (at dose 1 the constant seed scored -84 dB on the
spectra against joint_newton's 21; a random positive seed scored 21.4). A component dead in both factors
is a degenerate stationary point (zero gradient), so the re-seed escapes a saddle, not a KKT violation. The
shifted step is the inadmissible-zero offset of Chi & Kolda (2012) and the modified update of Lin (2007) in
translated form, and the extrapolated loop follows Ang & Gillis (2019) and the Richardson-Lucy acceleration of
Biggs & Andrews (1997); neither is new here (reading list, section 2). Nesterov extrapolation with
function-value restart cut sweeps 17x on the demo (6580 -> 380) at 1.3x the cost per sweep, 13x in wall
clock, lifting it from 40-70x slower than joint_newton to parity. The extrapolated loop returns the best
plain iterate and stops when the best has failed to improve by `rel_tol` per sweep over two consecutive
checks (the momentum loss is not monotone). It converges sublinearly, so `rel_tol` buys less than for
joint_newton: on the demo in float32, 1e-6 stops 275 sweeps in, 0.009% above the joint optimum (0.5 s);
1e-7 at 465 sweeps and 0.001%; 1e-8 at 660 sweeps and 0.0001% (1.3 s). A 2026-09-03 fix: `nnal_factorization`
tested `update is multiplicative_update` after `torch.compile` had replaced `update` with its wrapper, so
every compiled multiplicative solve silently ran the plain sweeps and stopped 0.06% above the optimum at
1.6x the sweeps; the compiled and eager updates themselves agree to 3e-7 per sweep.

**quadratic (IRLS).** Relinearized weighted least squares on the true NNAL. It replaced an update that
minimized `(1/2) sum T (X + log T)^2`, which converges somewhere other than the NNAL minimum and at low dose
discards a large fraction of the data (17.7% of entries at dose 3). Gauss-Newton style: single steps are not
guaranteed to decrease the loss.

**lbfgsb (added 2026-09-08).** scipy's L-BFGS-B over both factors at once under W, H >= 0, on the package's own
gradient: the generic bound-constrained baseline, and the solver generalized CP uses for the same Poisson log-link
loss (Hong, Kolda & Duersch 2020). `rel_tol` maps onto ftol. On the 4k phantom (laptop GPU): with 10 correction
pairs it stalls 6e-4 above the joint optimum at dose 3 even at ftol 1e-9 (623 iterations; the relative-decrease
test fires on a slow crawl), with 20 pairs it reaches the optimum to 3e-7 in 1152 iterations and 13.5 s against
joint Newton's 45 steps and 1.5 s; at dose 100, 588 iterations and 6.8 s against 20 steps and 0.4 s (m = 10
suffices there). Default memory 20. H100 (2026-09-08, dose 3, seed 129, ftol 1e-9, m = 20): 4.5 s at 4k pixels,
where it converges to joint Newton's loss (joint: 0.20 s), then 47 / 288 / 1880 s at 65k / 262k / 1M against joint
Newton's 1.2 / 3.7 / 14.1 s (40-133x), and from 16k pixels up it stops short: spectral SNR 27.6 / 30.1 / 30.5 /
30.4 dB against the MLE's 31.9 / 37.0 / 40.4 / 41.6 (loss gap +1.5e-3 at 1M), the relative-decrease test firing
on a slow crawl. Its wall clock is therefore a lower bound on the time to the MLE, and it grows faster than
linearly (log-log slope 1.33 above 65k) from the CPU-side vector work on 3M parameters and the transfers.

**Wall clock versus pixel count (H100 80 GB, 2026-09-08).** Dose 3, seed 129, rank 3, K = 1200; joint / multiplicative /
block at rel_tol 1e-8, L-BFGS-B at 1e-9; medians of 3 order-reversed repeats (2 at 1M, 1 for joint at 1M: the job hit
its 1h50 walltime in the third repeat because of L-BFGS-B's 31-minute runs). Post-estimators are the increment on the
MLE they start from. Seconds per solve:

| method | 4k | 16k | 65k | 262k | 1M | us/px at 1M | vs joint at 1M |
|---|---|---|---|---|---|---|---|
| joint-Newton (MLE) | 0.20 | 0.36 | 1.21 | 3.66 | 14.1 | 13.4 | 1.0x |
| multiplicative | 0.38 | 0.39 | 1.65 | 3.51 | 12.3 | 11.7 | 0.9x (0.8 dB short on spectra) |
| block-Newton | 1.35 | 3.02 | 11.1 | 46.9 | 206 | 197 | 14.6x |
| L-BFGS-B | 4.48 | 8.70 | 46.7 | 288 | 1880 | 1793 | 134x (not converged from 16k up) |
| + unconstrained spectra | 0.18 | 0.20 | 0.69 | 2.04 | 9.2 | 8.8 | 0.7x |
| + support selection | 0.39 | 0.96 | 2.76 | 11.1 | 45.6 | 43.5 | 3.2x |
| + pure-pixel gauge | 0.08 | 0.18 | 0.62 | 2.34 | 28.5 | 27.2 | 2.0x |
| L2 (scipy NMF, 14 CPU cores) | 4.47 | 20.2 | 81.0 | 136 | -- | 518 at 262k | 37x at 262k |

Joint Newton's per-pixel cost falls from 49.7 us at 4k to 18.4 at 65k, 14.0 at 262k and 13.4 at 1M: linear in pixel
count from 65k up (log-log slope 0.89 over that range), fixed overheads below. Peak memory 61.4 GB at 1M for the
Newton solvers, 36 GB for the multiplicative update. The gauge fix's W re-solve jumps from 2.3 s at 262k to 28.5 s at
1M (slope 1.38): the re-solve from the remixed start converges slowly at scale and is worth a look. L2 was not run at
1M (136 s at 262k on the CPU). Data: sweep_out/pscaling_h100.json in the scratch tree; Slurm job 16050171.

**Dose sweep on the H100 (2026-09-08).** 33 doses (8/decade, 1 to 1e4) x 10 seeds at 4k pixels, 8 methods, 37 min.
L-BFGS-B matches the MLE on spectra and maps at every dose (it converges at 4k); the pure-pixel gauge fix adds +5 to
+6 dB to the maps at every dose above ~3 (dose 100: 22.3 -> 27.7; 1e4: 42.2 -> 47.9), support selection + gauge is
best above dose 100 by a few tenths; the spectra remedies do not help at 4k, below the ~65k crossover. Per-solve
seconds: L2 2.9 (CPU), MLE 0.06, multiplicative 0.56, L-BFGS-B 1.44, + unconstrained 0.19, + support selection
0.40, + gauge 0.13, support selection + gauge 1.08. Slurm job 16049816.

**Stopping rule.** `rel_tol` means the relative loss change per step (float64 sum) for every method, with a KKT
fallback `gnorm <= max(rel_tol^2, 100 eps) gnorm0` for data a rank-R model fits exactly (the shifted loss then
goes to zero and its relative change stays O(1)). A KKT rule alone is relative to the gradient at the start, so
a better initialization would make the same tolerance stricter (at dose 100 the default init ran 4x longer
than a poor one for an identical loss). Unseeded batching made step counts non-reproducible (318 vs 565 on the
same data); the batched path seeds its pixel permutation.

## 4. Streaming (`stream_factorization`)

H is fitted on a subsample of the leading chunks with joint_newton; each full pass solves W per chunk with H
fixed and accumulates `W^T G`, the Khatri-Rao Hessian triangle and the per-bin loss in float64; one exact
Newton step on H follows, with a second pass for a four-point line search. A KKT residual
`||P(grad_H)|| / ||W^T T||` is reported per pass; the loss rule alone cannot tell a converged H from one sitting
in the line search's precision plateau. On the H100: ~21 s per pass on 10M pixels, GPU peak 34 GB with
524288-pixel chunks (64.7 GB with 1M chunks).

## 5. Bias of the maximum-likelihood spectra

The spectra H are shared by every pixel and estimated jointly with R nuisance coefficients per pixel, each
pixel carrying a fixed amount of information (Neyman-Scott). Measured at dose 3, rank 3 (spectral SNR):

| pixels | MLE (1e-8) | early stop (1e-6) |
|---|---|---|
| 262k | 40.4 | 41.2 |
| 524k | 41.1 | 42.3 |
| 1M | 41.7 | 43.2 |
| 10M (streaming) | 43.8 | -- |

The converged estimate gains 0.66 dB per doubling instead of 3; float64 reproduces float32; converging further
lowers the SNR while lowering the loss; a truth-started solve reaches the same optimum at K = 1200 on one seed
(not evidence of a unique MLE: at K = 300 two minima exist and the deeper one is the worse estimate; a
multi-start landscape study is still to be done). A Monte Carlo of the profile score at the true H
showed the bias is the truncation of pixel coefficients at zero (dropping W >= 0 removes it entirely, z = 4.9
-> 1.1), not the smooth O(1/m) term: Cox-Reid and Barndorff-Nielsen adjustments remove none of it, and a
parametric bootstrap of the score underestimates it because the fitted W has far fewer exact zeros than the
truth (`bias_corrected_spectra` is kept for reference only: +0.02 to +0.05 dB at 5 to 84x the cost).
Both failures are predicted by boundary theory, not discovered here: the modified profile likelihoods are
interior-mode Laplace approximations, invalid when the nuisance mode sits on a bound (Erkanli 1994), and the
parametric bootstrap is inconsistent for a parameter on the boundary (Andrews 2000). The per-pixel mechanism is
the projected-normal limit of a constrained estimate (Self & Liang 1987; Geyer 1994; Andrews 1999), and the
same clipping bias is known in constrained Poisson reconstruction (NEG-ML; Lim, Dewaraja & Fessler 2018). What
is new is the identification of truncation as the dominant incidental-parameter term of a jointly estimated
shared factor in a constrained factorization, with the consequence that the standard correction toolkit aims
at the wrong term (reading list, section 7).

**Remedies that work.**
- `unconstrained_spectra`: estimate H with the bound on W dropped, then re-solve W >= 0. Spectra 40.4 -> 43.2
  (262k), 41.1 -> 45.8 (524k), 41.7 -> 49.0 dB (1M), empirically restoring the 3 dB per doubling slope over
  262k-1M pixels on the phantom (the smooth O(R/K) Neyman-Scott term remains, so this is not a consistency
  statement); 0.6x the MLE's time; loses 0.7 dB at 4k pixels where the constraint's variance reduction still
  dominates (crossover near 65k at dose 3, later at lower dose): the implicit regularisation of a sign
  constraint (Slawski & Hein 2013; Meinshausen 2013). The H step with W free is a semi-NMF fit (Ding, Li &
  Jordan 2010) and the relax-where-it-truncates logic of NEG-ML in PET; the estimator is not new, its
  incidental-parameter justification and the measured crossover are. On 10M pixels streaming: 43.8 -> 49.0 dB
  after six passes, still improving.
- `support_selected_spectra`: penalised-likelihood choice of each pixel's material subset (all 2^R - 1), then a
  joint refit with the supports fixed. At 65k: penalty 1 (AIC) +0.79 dB spectra / +0.08 maps; 0.5 log K (BIC)
  +0.95 / +0.23; 2 log K +1.04 / +0.44 (default), exact support in 59% of pixels, 0.8x the MLE's time; also
  reduces the gauge mixing of the fitted rows (condition 49 -> 15). Iterating select/refit degenerates (round 2:
  -0.6 dB, round 3: -2.6 dB); one round is the optimum. Per-pixel subset selection is MESMA / ISMA (Roberts et
  al. 1998; Rogge et al. 2006) and exact sparse NNLS by enumeration (Cohen & Gillis 2019; Nadisic et al. 2020),
  and the refit is post-selection refitting; new here are the empty set in the penalised choice, the single
  refit, and the debiasing purpose. BIC and 2 log K gave the same result, so the penalty is not load-bearing.

**Maps are gauge-limited.** With the fitted spectra rotated into the true gauge, maps reach 10.8/11.5 dB at dose
3 and 20.9/22.2 at dose 30 (oracle ceilings 11.6/12.3 and 22.3/23.3) for every estimator alike, against 8.4-8.8
and 17.1-17.5 as fitted: 2.4-3.7 dB recoverable by a data-driven gauge criterion. "Gauge" here is a synonym for
what chemometrics has called the rotational ambiguity of bilinear models and the set (area) of feasible
solutions since Lawton & Sylvestre (1971): Abdollahi & Tauler (2011), Rajko (2009), Neymeyr & Sawall (2018);
in NMF theory Laurberg et al. (2008), Huang, Sidiropoulos & Swami (2014), Fu et al. (2018/2019). None of that
is new here. The measurement that is new is the oracle-rotation attribution: three estimators share one
post-rotation ceiling within 0.1 dB of the true-spectra oracle, so the whole deficit is mixing and none is
subspace, and a +7.3 dB spectral gain moved the maps only +0.15 dB.

**Gauge fix (`pure_pixel_gauge`, 2026-09-04).** The likelihood does not identify the gauge (the textbook
non-uniqueness of NMF: nonnegativity confines the mixing A to a polytope and the solver stops wherever its path
ends there). The remedy is not new either: Chowdhury et al. (ICIP 2023; IEEE Trans. Comput. Imaging 11, 2025)
already cluster the coefficient vectors of an NMF subspace fit of Bragg-edge neutron data, take the cluster
means as the mixing, remix the spectra and hold them fixed for the decomposition, under a one-material-per-voxel
assumption; the same construction is K-P-Means (Xu, Li, Wong & Peng 2014) and the vertex hunting of GeoNMF and
Topic-SCORE. `pure_pixel_gauge` moves it to the pixel domain of the exact-rank Poisson fit with a
likelihood-ratio material test and intensity-weighted k-means. In the MLE basis the true axes had
L1 rows like (0.48, 0.11, 0.41) at dose 3 and (0.07, 0.01, 0.92) at dose 30, so a pure pixel has no dominant
coefficient and a share threshold finds nothing (6 / 0 / 104 pixels at dose 3; one component dominated every pixel
at dose 30); support-selected single-material pixels gave axis error 0.19 and worse maps. Clustering instead:
intensity-weighted k-means (k-means++, 8 seeds) on the L1-normalised coefficient rows of the pixels with material
(likelihood ratio against X = 0 above 2 log K), axes = cluster means, H := A H, W >= 0 re-solved with H fixed (a
joint refit would return to the MLE gauge). It recovers the oracle: maps 8.35 -> 11.48 (oracle 11.47) at dose 3
and 17.12 -> 22.14 (oracle 22.18) at dose 30 at 65k pixels, axis error 0.012 / 0.0001, identical with or without
the weights, at 2 or 4 log K, and with constrained or unconstrained per-pixel means; the clustering takes 0.1-0.3 s
and the W re-solve is the only real cost. The loss rises by 7e-4 relative, the same as the oracle's, so the loss
cannot validate a gauge. Size dependence at dose 3 (seed 129): 4k pixels 7.90 -> 10.67 (oracle 10.97), 16k 8.29 ->
11.41 (11.51), 65k 8.35 -> 11.48 (11.47); at 4k and dose 30, 17.04 -> 22.08 (22.43). About a tenth of the weakly
attenuating aluminium pixels land in the other two clusters at every size, but the axis error still falls as
1/sqrt(pixels) (0.057 / 0.021 / 0.012), so the residual is the noise of the cluster means, not the misassignment.
The assumption is that every material has pure pixels; raw cluster means are biased inward when pixels are
mixed (Drumetz et al. 2020; K-P-Means purifies them), and the sufficiently-scattered / minimum-volume route
(Huang, Fu & Sidiropoulos 2016; Fu, Huang & Sidiropoulos 2018) is the alternative when a material has no pure
pixels, which is the case for two of three materials in the group's SNAP sample (the aluminium holder lies on
every ray through nickel and copper).

## 5b. Details removed from the code comments during the 2026-09-04 cleanup

Kept here so the pointers in the code lose nothing.
- Armijo floor: halving the floor halves the loss slop the streaming H step is allowed at large P (the noise
  ball in which H wandered at 43.8 dB on 9.4M pixels) for about half an extra loss evaluation per step.
- Compiled kernels: the GEMM-bound CG inner iteration does not benefit (1.07x). Compiled and eager agree
  bit-for-bit over a joint_newton solve (28 steps) and over the first 40 block_newton steps; over a 729-step
  block_newton run the fused reductions' different rounding eventually flips one active-set decision and the
  paths separate (max |W, H| difference 1.6e-2) but end at the same loss to 2e-8 relative with identical
  spectra. inductor reports too few SMs on the laptop card for its GEMM autotuning to apply. After the
  cleanup's dedup the compiled joint solver took 46 steps instead of 45 to the same loss (graph fusion
  differs once the direction code is a function), so the regression check allows +-2 steps on compiled and
  batched paths.
- joint_newton warm-up cost model: a block warm-up step costs about 1.4x a one-CG-iteration joint step.
- Joint Hessian-vector product: six GEMMs (dW@H, W@dH, ZdX@H^T, G@dH^T, W^T@ZdX, dW^T@G), not five as
  written elsewhere in these notes.
- NNDSVDa fill: the mean/100 fill converged in a reduced subspace, 0.6% worse; the c = 1 sqrt(mean) fill lands
  within 0.001% of the joint solver, and the multiplicative and joint solvers are indifferent to c across
  0.1..1; classic NNDSVDa on the badly floored matrix was 35x worse.
- Re-seeding dead components: the constant seed reaches the same loss as joint_newton to 0.3% at dose 3
  (at dose 1 it fails; see section 3).
- Nesterov restart cost: the cheap restart loss costs 0.5 ms against 4.3 ms for stable_nnal; checking with
  the full loss every sweep limited an earlier version to 4x. Float64 sums: the multiplicative methods once
  saw two identical consecutive float32 losses and stopped after two iterations.
- Batched path history: the previous version factored every batch, then factored the stacked spectra again
  with sklearn to reconcile them, one full solve per batch plus a host round trip.
- Streaming polish_dtype: on an H100, whose kernels here are memory-bound, float64 costs about 2x (and
  changed nothing; section 6).
- unconstrained_spectra small-P penalty: measured -0.8 dB at 4k (single seed, fp64) and -0.6..-0.8 across seeds
  in the grid, -0.7 at 16k, +0.3 at 65k; maps +0.15 dB.
- bias_corrected_spectra: the bootstrap correction diverged once (1M px, rank 4). The orthant adjustment
  removed 28% of the score bias with the right sign in the Monte Carlo (K = 300, dose 3) where Cox-Reid removed
  none; using the conditional Schur-complement curvature instead made it 2.2x too large with the wrong sign;
  a consistent version needs a differentiable bivariate/trivariate orthant probability.

## 6. Precision was not the cap
Phase 1b (H100): float64 equals float32 at 262k and 524k (40.35 vs 40.39; 41.02 vs 41.07 dB). Phase 2: float64
polish on 10M pixels reproduced every digit of the float32 run.

## 7. Large rank
On the three-material phantom, R = 10/30/100 free components: spectral SNR 32.1 -> 31.9 -> 30.6 -> 20.0 at 16k
pixels, step counts 60 -> 342 -> 600 (cap); the direct-fit map metric inflates with R while the coupled metric
collapses; the R = 30 and 100 runs hit the step cap and 3.9 of 4 GB, so the collapse is a solver-plus-conditioning
result, not a converged-estimator property. An unstructured factorization cannot learn a dictionary of
near-collinear dilated spectra (adjacent cosine 0.9997): material accuracy at chance from an NNDSVDa start, as
the shift-free controls of AgileFD (Suram et al. 2017) show for diffraction; structured models that learn a base
pattern and its dilation (AgileFD; StretchedNMF, Gu et al. 2024) are the untested alternative, so "cannot be
learned from scratch" holds only for the unstructured factorization. With a known dictionary and per-pixel
greedy nonnegative selection, material accuracy 0.92-0.98 and strain resolution ~0.25% dilation at dose 3,
~0.12% at dose 30, an order of magnitude coarser than multi-edge cross-correlation at conventional counts
(~90 microstrain, Ramadhan et al. 2019): a low-dose characterisation, not a new capability. Approximately
known spectra (texture, impurity) are recoverable in shape to 37-40 dB by one round of base-spectrum
refinement; absolute per-material scale is not identifiable from the data (H W is), so maps need a reference.

## 8. Novelty assessment (2026-09-06)
An adversarial literature review (ten scouts, ten refuters, four reviewer lenses; web-verified except where the
reading list says "verify") found close prior art for every method in this package and returned "partially
refuted" for every candidate contribution. Genuinely new in this setting, as measurements rather than methods:
(1) the shared-spectra bias of the constrained Poisson factorization is truncation-dominated, the standard
incidental-parameter corrections aim at the wrong term, and dropping the bound in the H step restores the
pixel-count trend (section 5); (2) the map deficit is mixing, not subspace, by the oracle-rotation attribution
across estimators (section 5); (3) the first Poisson-transmission factorization of Bragg-edge data at 1e7
pixels, with its cost model (section 4). To be positioned as instances of published work: the model (Poisson
exponential-family PCA with a log link; generalized CP lists this loss and solves it all at once with L-BFGS-B),
the joint Newton-CG (Sorber et al. 2013; Vandecappelle et al. 2021; Hansen, Plantenga & Kolda 2015), the
extrapolated multiplicative update (Lin 2007; Chi & Kolda 2012; Ang & Gillis 2019), the gauge fix (Chowdhury
et al. 2023/2025), the streaming (one-step polishing of GLMs; out-of-core NMF), the strain dictionary (AgileFD;
Balke et al. 2021 in this group) and the numerics (an appendix). Missing before publication: measured data
(SNAP; public IMAT sets), external baselines (all-at-once L-BFGS-B, variable projection, HALS-KL and Newton
KL-NMF, the AMD pipeline, SPA/VCA/min-vol NMF, edge fitting for strain), replicated seeds with intervals, a
proposition for the truncation bias, mixed-pixel and no-pure-pixel phantoms, instrument physics in the forward
model, standard metrics in physical units, and a reproducibility package. The full assessment with the per-claim
prior art is recorded separately as a slide deck.

## 9. Hybrid Bragg spectral model (`mbirtorch.hsnt.bragg`, branch `hybrid-spectra`, 2026-09-20)

The bilinear model `X = W H` does not identify the gauge: any invertible remix `(W A^-1, A H)` inside the nonnegative
polytope has the same likelihood, so the MLE spectra are mixtures and the maps are 5-6 dB below the known-spectra
ceiling (sections 5 and 8). The hybrid model removes the ambiguity by giving each crystalline component a physical
form,

    mu_r(lambda) = (lambda / lambda_ref)^2 * sum_hkl h_hkl * S(lambda - 2 d_hkl(a_r)) + c0 + c_abs * lambda + spline(lambda),

with `S` a smoothed step (Bragg edge: the coherent elastic cross-section loses the (hkl) family above `2 d_hkl`),
`d_hkl = a / sqrt(h^2 + k^2 + l^2)` for a cubic lattice, heights `h`, `c0`, `c_abs` nonnegative and an 8-knot cubic
B-spline residual with a ridge for whatever the physics above leaves out. The edge positions of one component are a
rigid pattern set by `(type, a)`, and a mixture of two lattices has edges of both, so rows cannot be remixed without
leaving the model: the gauge is fixed by the data. Nothing is assumed about the materials (user rule): the lattice
type (fcc, bcc, diamond) and parameter of every component are discovered.

**Discovery (`warm_start`).** For a fixed `(type, a)` the model is linear in its coefficients, so its spectra form a
subspace `D(a)`. The pure spectrum of a material lies, up to the model's approximation error, in the intersection of
`D(a)` with the row space of the bilinear MLE. Over a geometric grid of `a` in [2, 7] A and every type, the closest
pair of directions of the two spaces is the top canonical correlation (an `n x n` Gram eigenproblem per `a`, batched;
~2 s on the CPU for 3 x 1250 values). Local maxima are refined by golden section on sigma(a) (the direction is only
right at the exact `a`; a 0.02% grid error costs 1e-3 in fit rms). Candidates are scored by a BIC on the fit with
NONNEGATIVE heights to their own direction (the unconstrained correlation cannot tell a lattice from one whose edge set
contains it: parameter 2a, `sqrt2 a` across bcc/fcc, a superset type). Two more rules were needed: an alias whose
extra edges are insignificant (height / standard error < 3) or coincide with another candidate's edges is dropped (a
superset lattice can hold two materials' edges at once and then wins on correlation, as happened on the synthetic
test), and a candidate sharing most of its edges with a chosen one is skipped. A component whose fit rms exceeds 4x
the best (and 0.5%) is left nonparametric. Weak-edge materials remain ambiguous at low dose: fcc `a` and bcc
`a / sqrt2` share every even-index edge, so Al at dose 3 comes out as its bcc alias (spectrum still 14-20 dB).

**Solver (`hybrid_factorization`).** Alternating: (1) golden-section search on each `a` and an exact projected Newton
step on all linear coefficients against the convex quadratic model of the loss in `H` with `W` fixed (the block
solver's per-bin Newton statistics; a ~100-variable bounded QP via Cholesky + BVLS), Armijo-checked; (2) the convex
per-pixel `W` solve; (3) a Levenberg-damped Newton step on the PROFILED objective `L*(x) = min_W L(W, H(x))`, whose
Hessian is the Schur complement of the joint Hessian over the per-pixel 3x3 map blocks (accumulated in pixel chunks)
-- this is what moves along the valley of near-equivalent factorizations that the alternating moves crawl along
(from a poor start: 40 outer iterations and still +47 loss units above the truth; with the profiled step: 4-6). The
undamped profiled step from a poor start predicted a decrease larger than the whole loss (indefinite joint Hessian);
damping by the actual/predicted ratio fixed it. Free rows take one block Newton step per outer iteration (slow: 30
iterations when present). The bilinear maps are carried into the new gauge as the initial `W`.

**Measurements** (`claude_scratch/nnal_work/hybrid/exp4_results.json`; blind, natural order, scale-only fit per
component, seed 129, laptop GPU). Slab 64x64, dose 300: lattices fcc 3.6217 / 3.5230 / 4.0733 A (truth 3.6218 /
3.5231 / 4.0732); spectra 42.5 / 45.1 / 41.7 dB, against 25 / 3 / 6 for the blind bilinear rows (which need the
oracle gauge for 50 / 48 / 42); maps 34.7 / 30.0 / 19.9 against 32.3 / 27.5 / 20.6 for the bilinear WITH the oracle
gauge and 38.4 / 34.5 / 27.0 with the true spectra. Dose 30: spectra 37.8 / 40.6 / 33.5 (bilinear oracle gauge
39.6 / 38.4 / 31.3), maps 24.8 / 20.0 / 9.5 (gauge 22.6 / 17.7 / 10.8; known spectra 28.1 / 24.1 / 16.0). Spheres
(mixed pixels, 4 views), dose 300: maps 33.6 / 28.6 / 18.9 (gauge 31.8 / 26.4 / 20.0; known 34.8 / 30.4 / 23.5).
The hybrid loss sits at the truth's (8205 vs 8202; bilinear 8183): the bilinear overfits by exactly its gauge
freedom. Spectrum accuracy is capped by the model's approximation of the phantom's basis rows (43-46 dB on a direct
fit), not by noise. A fourth, non-crystalline component with a resonance line: the Bragg rows stay at 42-46 dB, the
free row's map reaches the bilinear-gauge level (16 dB) but its spectrum is not identified (its smooth part rotates
with the Bragg rows' smooth parts); the automatic crystalline/non-crystalline decision works at dose 300 and accepts
a spurious lattice at dose 30. Runtime: 27 s against 2.8 s for the bilinear MLE at 37k pixels (5 s discovery + 6
outer x 3.6 s, three `W` solves per iteration), 8 s against 0.7 s at 4k.

**What the results mean.** The material-agnostic Bragg parametrisation identifies spectra AND maps in natural order
at or above the level the bilinear model reaches only with an oracle gauge, on pure and mixed pixels, at 10x the
cost. The remaining gap to the known-spectra map ceiling (3-7 dB) is the soft rotation of the smooth parts through
the spline (a looser spline lowers the loss and worsens the maps; the 8-knot / ridge 1 default was tuned on this
phantom). Not done: polychromatic resolution kernel, hexagonal lattices, structure-factor priors (deliberately
excluded), a profiled step for free rows, real Ni-cylinder data.

### 9.1 Instrument resolution (2026-09-20, later)

Real edges are smeared by the time-of-flight resolution. The step in `BraggSpectrum` is now `edge_profile`: an
exponentially modified Gaussian kernel (Gaussian width sigma, exponential tail of decay tau toward longer wavelength,
the moderator's slow decay; Santisteban et al. 2001), both proportional to the edge wavelength as for a time-of-flight
instrument, with `(sigma / lam, tau / lam)` shared by every component (an instrument property) and fitted by
golden-section search on the same quadratic model as the lattice parameters (`fit_resolution=True`; `warm_start`
seeds the Gaussian width from the chosen directions). Two identifiability facts shaped the parametrisation. A tail
proportional to wavelength shifts every edge by the same fraction, exactly like a lattice dilation, so an uncentred
tail is not identifiable from `a`: on the phantom the tail collapsed to zero and `a` came out 0.5% high. Centring
the profile on the kernel's mean is not enough either: a wider symmetric Gaussian placed at the kernel's median
mimics the skewed profile, and the coordinate search on tau never leaves zero (synthetic test: sigma 0.0061 for
0.004, tau 0, a 0.4% low). Centred on the kernel's MEDIAN (`emg_median`, tabulated once by bisection and
interpolated) the lattice parameter is the half-height position of the smeared edge, the quantity an edge fit locks
onto, tau only sets the asymmetry about it, and the coordinate search converges: synthetic (0.0041, 0.0055) for
(0.004, 0.006), a within 4e-4 relative. Consequence to keep in mind: with an uncalibrated kernel `a` is the
half-height position; converting it to 2 d_hkl needs the kernel's median offset (a calibration with a reference
sample); relative strain between regions is unaffected.

Phantom rows broadened with sigma / lam = 0.004, tau / lam = 0.006 (4.8 and 7.1 bins at 3 A; blurred vs sharp basis
31-33 dB), slab 64x64, blind:

    dose 300   sharp-edge model:  loss +145 above the truth, a biased (3.5200 / 3.6209 / 4.0858 for 3.5231 / 3.6218 / 4.0732),
                                  spectra 25 / 33 / 33 dB, maps 30 / 15 / 4
               fitted resolution: (0.0039, 0.0057); loss at the truth's; a 3.5225 / 3.6223 / 4.0734;
                                  spectra 46 / 51 / 43, maps 33 / 27 / 18  (true kernel fixed: 47 / 52 / 44, 33 / 28 / 18)
    dose 30    sharp 28 / 32 / 31, maps 22 / 13 / 5  ->  fitted (0.0040, 0.0053): 37 / 41 / 33, maps 23 / 18 / 9 (= true kernel)

The kernel acts on the attenuation (the edge steps are smeared before the exponential). The measurement smears the
TRANSMISSION, and the two differ at second order in the edge jump; for this kernel on the phantom's nickel row the
chi-square excess per pixel at dose 300 is 0.12 / 1.25 / 9.0 (of 1200 bins) at X_max 0.5 / 1.0 / 2.1, so the
attenuation-domain kernel is adequate to X ~ 1 and a transmission-domain operator (fine grid, kernel after the
exponential, kernel-aware Newton statistics) is the next step for thick samples. Not modelled: time offsets constant
in wavelength, kernel shapes beyond the exponentially modified Gaussian, wavelength-dependent widths beyond
proportional.

### 9.2 Runtime: why 10x, and what remains

The estimate before implementation was 2-3x the bilinear solve. The first implementation measured 10x (27 s vs
2.6 s at 37k pixels on the laptop). Per-phase profiling attributed it as follows.

1. 57% in the map solves. Three per outer iteration where one suffices (the profiled step carries the maps'
   response, so no solve is needed between the coefficient step and it), and each took 6-7 Newton steps at 0.41 s
   instead of 2 at 0.11 s. The cause was a bug in `_two_metric_direction` (shared by every solver): a bound-adjacent
   component with an inward gradient was solved jointly in the Newton system and then overwritten by its scaled
   gradient, leaving the other components' moves, which assumed the joint solution, dangling. With physically
   normalised spectra the smooth parts of Cu and Al are nearly collinear, so for 109 pixels per step the direction
   raised the loss by 1.05 against a predicted decrease of 0.003 and the elementwise line search backtracked to its
   cap (8 float64 loss passes) on every step. Inward components are now kept out of the Newton system. The bilinear
   solvers see this rarely (their arbitrary gauge is better conditioned): the regression harness moves by 5 of
   45-357 steps and up to 5e-6 relative in loss (baseline not regenerated).
2. Discovery: 5.5 s of Python-level loops (650 single-parameter scans, 25 least-squares refits with 9-point grids).
   Batched golden section across peaks and single-solve scoring: 3.5-4 s, a fixed cost (1.8 s is the coarse scan of
   three lattice types on the CPU).
3. float64 products over pixels x bins in the profiled system, on a consumer GPU with 1/32-rate float64: products in
   the data's dtype with float64 sums, 0.48 -> 0.11 s per call.

Now 13.6 s vs 2.7 s (5x) at 37k pixels, 6.2 vs 0.7 at 4k: discovery 4 s + 6 outer x 1.6 s. The remaining excess over
the estimate is the fixed discovery cost (30% at 37k, amortised at scale), ~7 float64 loss evaluations per iteration
(0.3 s), map solves of 4-9 warm-started steps because every iteration moves H substantially, and ~200 small bounded
QPs per iteration for the lattice and resolution searches (0.9 s of CPU BVLS). The estimate assumed one Newton-step
equivalent per iteration; the model is fine, the implementation still spends about four.

### 9.3 Toward strain (what the model would and would not give)

A Bragg edge at 2 d_hkl records the spacing of planes NORMAL to the beam, so a transmission spectrum measures the
normal strain along the ray, averaged along the ray with the material's attenuation as the weight. The hybrid model
as written has one lattice parameter per material for the whole image; strain needs a per-pixel dilation of that
material's spectrum, `mu_r(lam / (1 + eps_pr))` (the strain track's parametrisation), one parameter per pixel and
material on top of the maps, plus a per-pixel broadening for strain gradients along the ray. Each projection then
yields the longitudinal ray transform of the strain tensor (Lionheart and Withers 2015), whose null space is the
symmetrised gradients of displacement fields; recovering the 3-D tensor needs equilibrium or compatibility
constraints (Airy or Beltrami stress functions, Gaussian-process priors: Wensrich, Hendriks, Gregg et al.
2016-2020), many rotation axes, or a reduced assumption (axisymmetry, plane stress). Assumptions to track: cubic
lattice and uniform texture (per-material edge heights are global here; texture varies edge heights and, with
per-pixel heights, can mimic strain); a calibrated wavelength axis and resolution kernel for absolute d (relative
strain needs only their stability); composition and temperature also dilate the lattice; the attenuation-weighted
average mixes strain with the density map; and the edge shift for 1e-3 strain is 0.3 bins here, so strain precision
is count-limited (earlier estimate 0.18% / 0.06% per pixel at dose 3 / 30).

Wavelength-axis audit (same day): nothing in `bragg.py` assumes a spectral range; the reflection index range follows
from `(2 a / lam_min)^2` (a fixed `hmax = 8` missed edges below 1.5 A), bin-relative widths use the median bin width,
the spline knots span the data, and the phantom basis's axis lives in `simulate.material_basis_wavelengths`.
