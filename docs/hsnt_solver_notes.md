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

**mann_multiplicative.** Damped multiplicative update with the shifted step `V <- max((V + d) r - d, 0)`, whose
fixed-point set equals the KKT set for any d > 0 (so zeros can be resurrected: on the demo 5.5% of W started
at zero and never moved under the plain update; 1.77% of entries were zero with a negative gradient).
Random re-seeding of components dead in both factors (at dose 1 the constant seed scored -84 dB on the
spectra against joint_newton's 21; a random positive seed scored 21.4). Nesterov extrapolation with
function-value restart cut sweeps 17x on the demo (6580 -> 380) at 1.3x the cost per sweep, 13x in wall
clock, lifting it from 40-70x slower than joint_newton to parity.

**quadratic (IRLS).** Relinearized weighted least squares on the true NNAL. It replaced an update that
minimized `(1/2) sum T (X + log T)^2`, which converges somewhere other than the NNAL minimum and at low dose
discards a large fraction of the data (17.7% of entries at dose 3). Gauss-Newton style: single steps are not
guaranteed to decrease the loss.

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
lowers the SNR while lowering the loss; a truth-started solve reaches the same optimum (unique MLE at K = 1200),
and at K = 300 the deeper minimum is the worse estimate. A Monte Carlo of the profile score at the true H
showed the bias is the truncation of pixel coefficients at zero (dropping W >= 0 removes it entirely, z = 4.9
-> 1.1), not the smooth O(1/m) term: Cox-Reid and Barndorff-Nielsen adjustments remove none of it, and a
parametric bootstrap of the score underestimates it because the fitted W has far fewer exact zeros than the
truth (`bias_corrected_spectra` is kept for reference only: +0.02 to +0.05 dB at 5 to 84x the cost).

**Remedies that work.**
- `unconstrained_spectra`: estimate H with the bound on W dropped, then re-solve W >= 0. Spectra 40.4 -> 43.2
  (262k), 41.1 -> 45.8 (524k), 41.7 -> 49.0 dB (1M), restoring the sqrt(N) rate; 0.6x the MLE's time; loses
  0.7 dB at 4k pixels where the constraint's variance reduction still dominates (crossover near 65k at dose 3,
  later at lower dose). On 10M pixels streaming: 43.8 -> 49.0 dB after six passes, still improving.
- `support_selected_spectra`: penalised-likelihood choice of each pixel's material subset (all 2^R - 1), then a
  joint refit with the supports fixed. At 65k: penalty 1 (AIC) +0.79 dB spectra / +0.08 maps; 0.5 log K (BIC)
  +0.95 / +0.23; 2 log K +1.04 / +0.44 (default), exact support in 59% of pixels, 0.8x the MLE's time; also
  reduces the gauge mixing of the fitted rows (condition 49 -> 15). Iterating select/refit degenerates (round 2:
  -0.6 dB, round 3: -2.6 dB); one round is the optimum.

**Maps are gauge-limited.** With the fitted spectra rotated into the true gauge, maps reach 10.8/11.5 dB at dose
3 and 20.9/22.2 at dose 30 (oracle ceilings 11.6/12.3 and 22.3/23.3) for every estimator alike, against 8.4-8.8
and 17.1-17.5 as fitted: 2.4-3.7 dB recoverable by a data-driven gauge criterion, still open.

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
collapses. Learning a dictionary of near-collinear dilated spectra (adjacent cosine 0.9997) from scratch fails
outright (material accuracy at chance); with a known dictionary and per-pixel greedy nonnegative selection,
material accuracy 0.92-0.98 and strain resolution ~0.25% dilation at dose 3, ~0.12% at dose 30. Approximately
known spectra (texture, impurity) are recoverable in shape to 37-40 dB by one round of base-spectrum
refinement; absolute per-material scale is not identifiable from the data (H W is), so maps need a reference.
