import numpy as np

from .denoise import rehydrate


# -----------------------------------------------------------------------
# Noisy Hyperspectral Neutron Data Simulation Function (Ni, Cu, and Al)
# -----------------------------------------------------------------------


def generate_hyper_data(material_basis, num_angles=1, detector_rows=64, detector_columns=64, dosage_rate=300,
                        material_density=None, noisy=True, verbose=1):
    """
    Simulate noisy hyperspectral neutron attenuation data for :math:`N_m=3` materials (Ni, Cu, Al) and :math:`N_k` wavelength bins.

    Args:
        material_basis: ndarray of shape :math:`(N_m, N_k)`, where rows are material linear attenuation coefficient spectra.
        num_angles: Number of view angles :math:`(N_v)`. Defaults to 1.
        detector_rows: Number of rows in the detector :math:`(N_r)`. Defaults to 64.
        detector_columns: Number of columns in the detector :math:`(N_c)`. Defaults to 64.
        dosage_rate: Neutron dosage rate during hyperspectral data collection. Defaults to 300.
        material_density: Material density (vol. fraction) for Ni, Cu, and Al. Defaults to {"Ni": 0.2, "Cu": 0.2, "Al": 1.0}.
        noisy: Whether to generate noisy data. Defaults to True.
        verbose: Verbosity level. If 0, prints nothing; if 1, prints details; if >1, also generates plots. Defaults to 1.

    Returns:
        A list in the form [noisy_hyper_projection, angles, gt_hyper_projection].
            - noisy_hyper_projection: Simulated noisy hyperspectral data of shape :math:`(N_v, N_r, N_c, N_k)`.
            - angles: ndarray of view angles in radians.
            - gt_hyper_projection: Ground truth noiseless hyperspectral data of same shape.

    """
    # Ensure material_basis has exactly 3 rows
    if material_basis.shape[0] != 3:
        raise ValueError("material_basis must have exactly 3 rows (Ni, Cu, Al).")

    # Validate geometry and inputs
    if detector_rows < 3 or detector_columns < 2:
        raise ValueError("detector_rows must be ≥3 and detector_columns ≥2.")
    if dosage_rate <= 0:
        raise ValueError("dosage_rate must be positive.")

    # Handle default material_density and verify required keys
    if material_density is None:
        material_density = {"Ni": 0.2, "Cu": 0.2, "Al": 1.0}
    required = {"Ni", "Cu", "Al"}
    missing = required - set(material_density)
    if missing:
        raise KeyError(f"material_density missing keys: {sorted(missing)}")

    # Basic sanity on basis values
    if np.any(material_basis < 0):
        raise ValueError("material_basis should be non-negative attenuation coefficients.")

    # Set variable values
    epsilon = 1e-30
    number_of_materials = material_basis.shape[0]
    number_of_wavelengths = material_basis.shape[1]

    # Generate view angles
    angles = np.linspace(0, np.pi, num_angles)

    # Generate simulated projection data for 3 materials (Ni, Cu, and Al)
    height = detector_rows // 3
    width = detector_columns // 2
    # -(width // 2), not -width // 2: Python floors, so for odd width the latter
    # is one further from zero than width // 2, the range is asymmetric, and the
    # square root goes negative at one end (NaN -> zeroed thickness). Bit at 91x91.
    thickness = 20 * np.sqrt((width//2)**2 - np.linspace(-(width // 2), width // 2, width)**2)/ width
    material_projection = np.zeros((num_angles, detector_rows, detector_columns, number_of_materials), dtype=material_basis.dtype)
    material_projection[:, :height, width // 2:width + width // 2, 0] = material_density["Ni"] * thickness
    material_projection[:, 2 * height:, width // 2:width + width // 2, 1] = material_density["Cu"] * thickness
    material_projection[:, height:2 * height, width // 2:width + width // 2, 2] = material_density["Al"] * thickness

    # Generate noiseless hyperspectral projection data using rehydrate function
    gt_hyper_projection = rehydrate([material_projection, material_basis, 'attenuation'])

    # Generate noiseless hyperspectral open beam data using the given dosage rate
    noiseless_open_beam = dosage_rate * np.ones((detector_rows, detector_columns, number_of_wavelengths), dtype=material_basis.dtype)

    # Generate noiseless raw hyperspectral neutron counts
    noiseless_object_scan = np.exp(-gt_hyper_projection) * noiseless_open_beam
    noiseless_object_scan = np.nan_to_num(noiseless_object_scan, nan=0, posinf=0, neginf=0)

    if noisy:
        # Generate noisy neutron counts from Poisson distribution
        noisy_object_scan = np.random.poisson(noiseless_object_scan)
    else:
        # Do not generate noisy data
        noisy_object_scan = noiseless_object_scan

    # Generate noisy hyperspectral projection data
    ratio = noisy_object_scan / noiseless_open_beam
    ratio[ratio < epsilon] = epsilon
    noisy_hyper_projection = -np.log(ratio)

    if verbose >= 1:
        print("generate_hyper_data(): ")
        print("   -Shape of material_basis (linear attenuation coefficients for Ni, Cu, and Al):", material_basis.shape)
        print("   -Shape of material_projection (density of Ni, Cu, and Al):", material_projection.shape)
        print("   -Shape of hyperspectral data: ", noisy_hyper_projection.shape)

    if verbose > 1:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(material_basis.T)  # each column is a basis function
        plt.xlabel("wavelength index")
        plt.ylabel("linear attenuation ($cm^{-1}$)")
        plt.title("Material basis functions (Ni, Cu, Al)")
        plt.legend(["Ni", "Cu", "Al"])

    return [noisy_hyper_projection, angles, gt_hyper_projection]


# -----------------------------------------------------------------------
# Three spheres in a triangle, viewed within the plane of the triangle
# -----------------------------------------------------------------------
def generate_sphere_data(material_basis, num_angles=4, detector_rows=64, detector_columns=64, dosage_rate=300,
                         material_density=None, sphere_radius=None, triangle_radius=None, angles=None, noisy=True,
                         verbose=1):
    """
    Simulate hyperspectral neutron data for three solid spheres (Ni, Cu, Al) whose centres form an equilateral
    triangle, viewed by a parallel beam from directions within the plane of the triangle.

    The rotation axis is the detector-row axis, perpendicular to the triangle; the detector columns span the plane.
    Because every view lies in the plane, the spheres overlap in projection at most angles (fully for one pair at
    angle 0), so the phantom has mixed pixels as well as pure ones, unlike the slab phantom of
    :func:`generate_hyper_data`, whose pixels are all pure. Each pixel's areal density of a material is its volume
    fraction times the chord length through that sphere, with the sphere diameter scaled to the same 10 thickness
    units as the slabs' maximum, so attenuations are comparable between the two phantoms.

    Args:
        material_basis: ndarray of shape :math:`(3, N_k)`, rows Ni, Cu, Al linear attenuation spectra.
        num_angles: Number of views :math:`N_v`, equally spaced over :math:`[0, \\pi)` unless ``angles`` is given.
        detector_rows, detector_columns: Detector size; the spheres sit in the middle rows.
        dosage_rate: Open-beam counts per pixel and wavelength bin.
        material_density: Volume fractions for Ni, Cu, Al. Defaults to {"Ni": 0.2, "Cu": 0.2, "Al": 1.0}.
        sphere_radius: Sphere radius in pixels. Default 0.19 of the detector width.
        triangle_radius: Circumradius of the triangle of centres in pixels. Default 0.23 of the detector width, which
            keeps the spheres from intersecting in 3-D (side 0.4 of the width against a diameter of 0.38).
        angles: Optional array of view angles in radians, overriding ``num_angles``.
        noisy: Draw Poisson counts; otherwise return the noiseless data as the measurement.
        verbose: 0 silent, 1 prints shapes and the per-view overlap statistics.

    Returns:
        [noisy_hyper_projection, angles, gt_hyper_projection, material_projection]: the measured attenuation
        :math:`(N_v, N_r, N_c, N_k)`, the angles, the noiseless attenuation, and the ground-truth areal densities
        :math:`(N_v, N_r, N_c, 3)` (Ni, Cu, Al).
    """
    if material_basis.shape[0] != 3:
        raise ValueError("material_basis must have exactly 3 rows (Ni, Cu, Al).")
    if np.any(material_basis < 0):
        raise ValueError("material_basis should be non-negative attenuation coefficients.")
    if dosage_rate <= 0:
        raise ValueError("dosage_rate must be positive.")
    if material_density is None:
        material_density = {"Ni": 0.2, "Cu": 0.2, "Al": 1.0}
    missing = {"Ni", "Cu", "Al"} - set(material_density)
    if missing:
        raise KeyError(f"material_density missing keys: {sorted(missing)}")
    r = detector_columns * 0.19 if sphere_radius is None else float(sphere_radius)
    rho = detector_columns * 0.23 if triangle_radius is None else float(triangle_radius)
    if rho * np.sqrt(3) < 2 * r and verbose:
        print(f"generate_sphere_data(): spheres intersect in 3-D (side {rho * np.sqrt(3):.1f} < diameter {2 * r:.1f}); "
              "overlapping regions carry both materials")
    angles = np.linspace(0, np.pi, num_angles, endpoint=False) if angles is None else np.asarray(angles, dtype=float)
    # centres of the equilateral triangle in the plane (x, z), Ni at the top
    phis = np.deg2rad([90.0, 210.0, 330.0])
    centres = rho * np.stack([np.cos(phis), np.sin(phis)], 1)
    densities = np.array([material_density["Ni"], material_density["Cu"], material_density["Al"]], dtype=float)
    v = np.arange(detector_rows) - (detector_rows - 1) / 2.0            # along the rotation axis
    u = np.arange(detector_columns) - (detector_columns - 1) / 2.0      # in-plane detector coordinate
    material_projection = np.zeros((len(angles), detector_rows, detector_columns, 3), dtype=material_basis.dtype)
    for a, th in enumerate(angles):
        p = np.array([-np.sin(th), np.cos(th)])                         # in-plane unit vector across the beam
        for m in range(3):
            du = u[None, :] - centres[m] @ p
            chord2 = r * r - du ** 2 - v[:, None] ** 2
            thickness = 10.0 * np.sqrt(np.clip(chord2, 0, None)) / r          # 10 units across a diameter, as the slabs
            material_projection[a, :, :, m] = densities[m] * thickness
    gt_hyper_projection = rehydrate([material_projection, material_basis, 'attenuation'])
    noiseless_counts = np.nan_to_num(dosage_rate * np.exp(-gt_hyper_projection), nan=0, posinf=0, neginf=0)
    counts = np.random.poisson(noiseless_counts) if noisy else noiseless_counts
    ratio = counts / dosage_rate
    ratio[ratio < 1e-30] = 1e-30
    noisy_hyper_projection = -np.log(ratio)
    if verbose >= 1:
        print("generate_sphere_data(): ")
        print(f"   -sphere radius {r:.1f} px, triangle circumradius {rho:.1f} px, side {rho * np.sqrt(3):.1f} px, "
              f"{len(angles)} views at {np.round(np.rad2deg(angles), 1).tolist()} deg")
        print("   -Shape of material_projection (areal density of Ni, Cu, Al):", material_projection.shape)
        print("   -Shape of hyperspectral data: ", noisy_hyper_projection.shape)
        present = material_projection > 0
        for a in range(len(angles)):
            n = present[a].sum(-1); mat = n > 0
            print(f"   -view {a} ({np.rad2deg(angles[a]):5.1f} deg): {mat.mean():.1%} of pixels hold material; of those "
                  f"{(n[mat] == 1).mean():.0%} pure, {(n[mat] == 2).mean():.0%} two materials, {(n[mat] == 3).mean():.0%} three")
    return [noisy_hyper_projection, angles, gt_hyper_projection, material_projection]


def material_basis_wavelengths(num_bins, lam0=1.5099, step=0.0025196):
    """Wavelength (Angstrom) of each bin of the phantom's `material_basis.npy`, which stores spectra without an axis.

    The axis was calibrated from the nickel row (fcc, a = 3.5231 A): its edges at bins 26, 43, 100, 208, 244, 390,
    799 and 1016 match 2 a / sqrt(h^2 + k^2 + l^2) for the (531), (511)/(333), (422), (420), (331), (400), (222) and
    (311) families with a linear axis lam = lam0 + step * bin to 1.3 mA rms. Anything that gives the hybrid Bragg
    model a wavelength axis for phantom data should call this instead of repeating the constants.
    """
    return lam0 + step * np.arange(num_bins, dtype=np.float64)
