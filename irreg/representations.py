import numpy as np

from irreg.emsc import emsc_batched


def emsc_b(
    spectral_cube: np.ndarray,
    wavenumbers: np.ndarray,
    reference_spectrum: np.ndarray = None,
    batch_size: int = 512,
) -> np.ndarray:
    spectra = spectral_cube.reshape((-1, spectral_cube.shape[2]))
    _, coeffs = emsc_batched(
        spectra,
        wavenumbers,
        reference=reference_spectrum,
        poly_order=2,
        batch_size=batch_size,
        out_dtype="float",
    )
    coeffs = coeffs.reshape(spectral_cube.shape[:2] + (coeffs.shape[1],))

    b = coeffs[:, :, 0]

    return b


def linear_YCbCr(data: np.ndarray):
    """ITU-R BT.601"""
    # fmt: off
    return (0.299 * data[:, :, 0]) + \
           (0.587 * data[:, :, 1]) + \
           (0.114 * data[:, :, 2])
    # fmt: on
