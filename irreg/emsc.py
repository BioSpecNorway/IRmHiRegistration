from typing import Tuple

import numpy as np


def emsc(
    spectra: np.ndarray,
    wavenumbers: np.ndarray,
    reference: np.ndarray = None,
    order: int = 2,
    return_preprocessed_spectra: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess all spectra with EMSC
    :param spectra:
    :param wavenumbers:
    :param order:
    :return: coeffs
    """
    if reference is None:
        reference = np.mean(spectra, axis=0)
    reference = reference[:, np.newaxis]

    m0 = 2.0 / (wavenumbers[0] - wavenumbers[-1])
    c_coef = 0.5 * (wavenumbers[0] + wavenumbers[-1])
    transformed_wn = (wavenumbers - c_coef) * m0

    polynomial_columns = np.ones((len(wavenumbers), 1))
    for j in range(1, order + 1):
        polynomial_columns = np.concatenate(
            (polynomial_columns, (transformed_wn ** j)[:, np.newaxis]), axis=1
        )
    A = np.concatenate((reference, polynomial_columns), axis=1)

    """ we are going to solve A*X = B with OLS
    N - number of wns, M - number of spectra
    A has (N, order+2) shape
    X has (order+2, M) shape
    B has (N, M) shape

    A has the following structure
    (
        x1 1  wn1 .. wn1^(order)
        x2 1  wn2 .. wn2^(order)
        x3 1  wn3 .. wn3^(order)
        .. .. ..  .. ..
        xN 1  wnN    wnN^(order)
    )
    X has the following
    (
        b1  b2  .. aM                       -> a
        a1  a2  .. bM                       -> b
        d11 d12 .. d1M                      -> d1
        d21 d22 .. d2M                      -> d2
        ...
        d(order)1 d(order)2 .. d(order)M    -> d(order)
    )
    B is transposed matrix of input spectra
    We need to compute matrix X to compute vector coeffs (a, b, d1, d2, ..., dN)

    Then the solution for X is (A^T*A)^(-1)*A^T*B
    First we need to compute pseudo inversed matrix of A
    A_pseud_inv = (A^T*A)^(-1)*A^T
    A_pseud_inv has (order+2, N) shape
    """
    A_pseud_inv = np.dot(np.linalg.pinv(np.dot(A.T, A)), A.T)

    spectra_columns = spectra.T
    coeffs = np.dot(A_pseud_inv, spectra_columns)

    preprocessed_spectra = None
    if return_preprocessed_spectra:
        residues = spectra_columns - np.dot(A, coeffs)
        preprocessed_spectra = reference + residues / coeffs[0]
        preprocessed_spectra = preprocessed_spectra.T

    return preprocessed_spectra, coeffs.T


def emsc_batched(
    spectra: np.ndarray,
    wavenumbers: np.ndarray,
    reference: np.ndarray = None,
    poly_order: int = 2,
    return_coefs: bool = True,
    return_preprocessed_spectra: bool = False,
    batch_size: int = 512,
    out_dtype=None,
    emsc_dtype=np.float32,
):
    n_spectra = len(spectra)
    n_coeffs = 1 + (poly_order + 1)

    wavenumbers = wavenumbers.astype(emsc_dtype)

    if reference is None:
        reference = np.mean(spectra, axis=0, dtype=emsc_dtype)
    reference = reference.astype(emsc_dtype)

    if out_dtype is None:
        out_dtype = spectra.dtype

    preprocessed_spectra = None
    if return_preprocessed_spectra:
        preprocessed_spectra = np.zeros(spectra.shape, dtype=out_dtype)

    coeffs = None
    if return_coefs:
        coeffs = np.zeros((n_spectra, n_coeffs), dtype=out_dtype)

    for i in range(0, n_spectra, batch_size):
        preprocessed_spectra_batch, coeffs_batch = emsc(
            spectra[i : i + batch_size].astype(emsc_dtype),
            wavenumbers,
            reference=reference,
            order=poly_order,
            return_preprocessed_spectra=return_preprocessed_spectra,
        )
        if return_preprocessed_spectra:
            preprocessed_spectra[i : i + batch_size] = preprocessed_spectra_batch

        if return_coefs:
            coeffs[i : i + batch_size] = coeffs_batch

    return preprocessed_spectra, coeffs
