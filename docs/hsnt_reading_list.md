# Reading list for the techniques in the `mbirtorch.hsnt` package

Grouped by technique, with the part of the code each reference explains.

## 1. The loss: Poisson transmission likelihood (NNAL)
- Lange & Carson, *EM reconstruction algorithms for emission and transmission tomography*, J. Comput. Assist. Tomogr. 8 (1984). Origin of the transmission Poisson log-likelihood `sum exp(-x) + t x`.
- Erdogan & Fessler, *Monotonic algorithms for transmission tomography*, IEEE Trans. Med. Imaging 18 (1999). Surrogate / separable-quadratic treatment of this loss; background for `quadratic_update` and for the convexity of the per-pixel problem.
- Sauer & Bouman, *A local update strategy for iterative reconstruction from projections*, IEEE Trans. Signal Process. 41 (1993). Coordinate-descent lineage for transmission data.

## 2. Nonnegative factorization: model, multiplicative updates, initialization
- Lee & Seung, *Algorithms for non-negative matrix factorization*, NIPS 13 (2001). The multiplicative update generalized by `multiplicative_update`.
- Fevotte & Idier, *Algorithms for nonnegative matrix factorization with the beta-divergence*, Neural Computation 23 (2011). Majorization view of multiplicative updates for Poisson/KL losses; damping; why plain MU cannot resurrect zeros.
- O'Donoghue & Candes, *Adaptive restart for accelerated gradient schemes*, Found. Comput. Math. 15 (2015). Nesterov extrapolation with function-value restart (Mann update).
- Boutsidis & Gallopoulos, *SVD based initialization: a head start for nonnegative matrix factorization*, Pattern Recognition 41 (2008). NNDSVD / NNDSVDa (`nndsvda`).
- Halko, Martinsson & Tropp, *Finding structure with randomness*, SIAM Review 53 (2011). Randomized SVD inside the initializer.
- Gillis, *Nonnegative Matrix Factorization*, SIAM (2020). Uniqueness, separability, minimum volume, algorithms; the gauge freedom (W, H) -> (W A, A^-1 H).
- Fu, Huang, Sidiropoulos & Ma, *Nonnegative matrix factorization for signal and data analytics: identifiability, algorithms, and applications*, IEEE Signal Process. Mag. 36 (2019).

## 3. Projected Newton with bounds (`block_newton_step`, `_h_direction`)
- Bertsekas, *Projected Newton methods for optimization problems with simple constraints*, SIAM J. Control Optim. 20 (1982). Two-metric projection, epsilon-active set (`_ACTIVE_TOL`).
- Gafni & Bertsekas, *Two-metric projection methods for constrained optimization*, SIAM J. Control Optim. 22 (1984).
- Bertsekas, *Nonlinear Programming*, 3rd ed., Athena Scientific (2016), Sec. 3.3. Gradient projection, two-metric methods, Armijo along the projection arc.
- Kim, Sra & Dhillon, *Fast Newton-type methods for the least squares nonnegative matrix factorization problem*, SDM (2007).
- Lin, *Projected gradient methods for nonnegative matrix factorization*, Neural Computation 19 (2007).

## 4. Joint truncated Newton (`joint_newton_optimize`, `_joint_newton_pcg`)
- Nocedal & Wright, *Numerical Optimization*, 2nd ed., Springer (2006). Ch. 3 line search, Ch. 4 trust regions, Ch. 5 conjugate gradients, Ch. 7 inexact Newton / Newton-CG, Sec. 10.3 Levenberg-Marquardt damping.
- Nash, *A survey of truncated-Newton methods*, J. Comput. Appl. Math. 124 (2000).
- Eisenstat & Walker, *Choosing the forcing terms in an inexact Newton method*, SIAM J. Sci. Comput. 17 (1996). Adaptive CG tolerance.
- Pearlmutter, *Fast exact multiplication by the Hessian*, Neural Computation 6 (1994); Martens, *Deep learning via Hessian-free optimization*, ICML (2010). Matrix-free Hessian-vector products.

## 5. IRLS and majorization (`quadratic_update`)
- Green, *Iteratively reweighted least squares for maximum likelihood estimation, and some robust and resistant alternatives*, JRSS B 46 (1984).
- Hunter & Lange, *A tutorial on MM algorithms*, The American Statistician 58 (2004).

## 6. Floating point (fp64 accumulation, `_ARMIJO_FLOOR`)
- Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM (2002), Ch. 2-4. Rounding of long sums; random-walk vs systematic rounding.

## 7. Incidental parameters: bias of the ML spectra
- Neyman & Scott, *Consistent estimates based on partially consistent observations*, Econometrica 16 (1948).
- Lancaster, *The incidental parameter problem since 1948*, J. Econometrics 95 (2000). Survey.
- Kiefer & Wolfowitz, *Consistency of the maximum likelihood estimator in the presence of infinitely many incidental parameters*, Ann. Math. Statist. 27 (1956).
- Barndorff-Nielsen, *On a formula for the distribution of the maximum likelihood estimator*, Biometrika 70 (1983); Cox & Reid, *Parameter orthogonality and approximate conditional inference*, JRSS B 49 (1987); Severini, *An approximation to the modified profile likelihood function*, Biometrika 85 (1998). The `cox_reid` and `barndorff_nielsen` adjustments and why orthogonality matters.
- McCullagh & Tibshirani, *A simple method for the adjustment of profile likelihoods*, JRSS B 52 (1990). Adjusting the profile score by its expectation.
- Hahn & Newey, *Jackknife and analytical bias reduction for nonlinear panel models*, Econometrica 72 (2004); Kuk, *Asymptotically unbiased estimation in generalized linear models with random effects*, JRSS B 57 (1995). Bootstrap/jackknife bias corrections (`bias_corrected_spectra`, bootstrap).
- Firth, *Bias reduction of maximum likelihood estimates*, Biometrika 80 (1993).

## 8. Support selection and sparse nonnegative fitting (`support_selected_spectra`, strain greedy selector)
- Schwarz, *Estimating the dimension of a model*, Ann. Statist. 6 (1978) (BIC); Akaike (1974) (AIC).
- Bruckstein, Elad & Zibulevsky, *On the uniqueness of nonnegative sparse solutions to underdetermined systems of equations*, IEEE Trans. Inf. Theory 54 (2008).
- Pati, Rezaiifar & Krishnaprasad, *Orthogonal matching pursuit*, Asilomar (1993); Yaghoobi, Wu & Davies, *Fast non-negative orthogonal matching pursuit*, IEEE Signal Process. Lett. 22 (2015).

## 9. Streaming factorization (`stream_factorization`)
- Mairal, Bach, Ponce & Sapiro, *Online learning for matrix factorization and sparse coding*, JMLR 11 (2010). Sufficient-statistic accumulation for the shared factor.

## 10. Domain: energy-resolved neutron transmission and Bragg edges
- Santisteban, Edwards, Steuwer & Withers, *Time-of-flight neutron transmission diffraction*, J. Appl. Cryst. 34 (2001).
- Tremsin et al., *High-resolution strain mapping through time-of-flight neutron transmission diffraction with a microchannel plate neutron counting detector*, Strain 48 (2012).
- Bioucas-Dias et al., *Hyperspectral unmixing overview: geometrical, statistical, and sparse regression-based approaches*, IEEE JSTARS 5 (2012).

Suggested order: Nocedal & Wright ch. 3/5/7 and Bertsekas (1982) for the solver; Gillis (2020) on identifiability for the gauge; Lancaster (2000) then Cox & Reid (1987) for the bias story; Santisteban (2001) for the strain physics.
