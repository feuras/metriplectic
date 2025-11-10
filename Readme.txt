Metriplectic Numerical Test Suite

This suite provides minimal, self-contained numerical verifications of metriplectic structure, energy–entropy balance, and phase-blind invariance in 1D and 2D systems. Each script is NumPy-based, and replicable in a standard Python or Colab environment.

01_wave-dispersion_probe-fft_suite.py	Probe-based Fourier dispersion tests for the 2D wave equation.	Isotropic dispersion (ω² = c²k²), energy drift, anisotropy fits, damping decay, and time-reversal error.
02_metriplectic_equivalence_1d_periodic_kkt.py	1D finite-volume verification of metriplectic equivalence and inequalities.	Equality: ; Inequality: ; Fisher curvature diagnostic.
03_path-entropy_invariance_metriplectic_batch.py	2D high-rigor batch verification of path-integrated entropy invariance under reversible flows.	Tests  across multiple -amplitudes and random seeds; invariant total entropy to ≤10⁻⁴.
04_heat-identity_phase-blind_pinnedDC.py	1D exact-heat (λ=0) dissipation with DC-pinned mass.	 verified by midpoint rule; monotonic , positive , uniform equilibrium.
05_metriplectic_identity_phase-blind_pinnedDC.py	Dual experiment: pure heat and reversible + heat under exact DC pinning.	Both cases satisfy  to 10⁻⁵–10⁻⁶ relative error; phase-blind symmetry confirmed.
06_metriplectic_commuting-triangle.py	Final commuting-triangle suite linking reversible, dissipative, and constraint flows.	Verifies metriplectic commutativity and energy balance under mixed evolution.
