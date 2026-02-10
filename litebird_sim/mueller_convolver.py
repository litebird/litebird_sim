# fmt: off
# MR: This code is maintained externally, so I'm switching auto-formatting off
# to make updating from outside sources easier.
from dataclasses import dataclass
from typing import List, Optional

import ducc0
import numpy as np

__all__ = ["MuellerConvolver"]


def nalm(lmax: int, mmax: int) -> int:
    """Returns the number of alm coefficients for a given lmax, mmax."""
    return ((mmax + 1) * (mmax + 2)) // 2 + (mmax + 1) * (lmax - mmax)


# Adri 2020 A25/A35
# Conjugate by Yuya
def mueller_to_C(mueller: np.ndarray) -> np.ndarray:
    """
    Converts a 4x4 Mueller matrix to the coherence matrix basis (T, P, P*, V).
    """
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    T = np.zeros((4, 4), dtype=np.complex128)
    T[0, 0] = T[3, 3] = 1.0
    T[1, 1] = T[2, 1] = inv_sqrt2
    T[1, 2] = 1j * inv_sqrt2
    T[2, 2] = -1j * inv_sqrt2
    # C = T @ M @ T_dagger
    return np.conj(T.dot(mueller.dot(np.conj(T.T))))


@dataclass
class TruncatedBlm:
    blms: np.ndarray
    mmax: int


def truncate_blm(inp: np.ndarray, lmax: int, mmax: int, epsilon: float = 1e-10) -> List[Optional[TruncatedBlm]]:
    """
    Analyzes the beam components and truncates them at the highest significant m-mode.
    """
    limit = epsilon * np.max(np.abs(inp))
    out: List[Optional[TruncatedBlm]] = []
    for i in range(len(inp)):
        maxk = -1
        idx = 0
        for k in range(mmax + 1):
            # Check if this m-block has significant power
            block_len = lmax + 1 - k
            if np.max(np.abs(inp[i, :, idx : idx + block_len])) > limit:
                maxk = k
            idx += block_len
        if maxk == -1:
            out.append(None)
        else:
            out.append(
                TruncatedBlm(
                    blms=inp[i, :, : nalm(lmax, maxk)].copy(),
                    mmax=maxk,
                )
            )
    return out


class MuellerConvolver:
    """Class for computing convolutions between arbitrary beams and skies in the
    presence of an optical element with arbitrary Mueller matrix in front of the
    detector.

    Most of the expressions in this code are derived from
    Duivenvoorden et al. 2021, MNRAS 502, 4526
    (https://arxiv.org/abs/2012.10437)

    Harmonic Layout Convention
    --------------------------
    All spherical harmonic inputs (`slm`, `blm`) must strictly follow the
    **Healpix m-major ordering** convention, which is the same layout used by
    the `SphericalHarmonics` class in this library.

    - Indexing: The 1D array index for a given (l, m) is:
      ``idx = m * (2 * lmax + 1 - m) // 2 + l``
    - Ordering: The array is a concatenation of blocks of constant m:
      [ (m=0, l=0..lmax), (m=1, l=1..lmax), ..., (m=mmax, l=mmax..lmax) ]

    Parameters
    ----------
    lmax : int
        maximum l moment of the provided sky and beam a_lm
    kmax : int
        maximum m moment of the provided beam a_lm
    slm : numpy.ndarray((n_comp, n_slm), dtype=complex)
        input sky a_lm
        ncomp can be 1, 3, or 4, for T, TEB, TEBV components, respectively.
        The components have the a_lm format used by healpy
    blm : numpy.ndarray((n_comp, n_blm), dtype=complex)
        input beam a_lm
        ncomp can be 1, 3, or 4, for T, TEB, TEBV components, respectively.
        The components have the a_lm format used by healpy
    mueller : np.ndarray((4,4), dtype=np.float64)
        Mueller matrix of the optical element in front of the detector
    single_precision : bool
        if True, store internal data in single precision, else double precision
    epsilon : float
        desired accuracy for the interpolation; a typical value is 1e-4
    npoints : int
        total number of irregularly spaced points you want to use this object for
        (only used for performance fine-tuning)
    sigma_min, sigma_max: float
        minimum and maximum allowed oversampling factors
        1.2 <= sigma_min < sigma_max <= 2.5
    nthreads : int
        the number of threads to use for computation
    """
    def __init__(
        self,
        *,
        slm: np.ndarray,
        blm: np.ndarray,
        mueller: np.ndarray,
        lmax: int,
        kmax: int,
        single_precision: bool = True,
        epsilon: float = 1e-4,
        npoints: int = 1000000000,
        sigma_min: float = 1.2,
        sigma_max: float = 2.5,
        nthreads: int = 1,
    ):
        self._ftype = np.float32 if single_precision else np.float64
        self._ctype = np.complex64 if single_precision else np.complex128
        self._slm = slm.astype(self._ctype, copy=False)
        self._lmax = lmax
        self._kmax = kmax

        # Prepare the effective beams (mixing physical beam + Mueller)
        # Result shape: (5, ncomp, n_alm)
        tmp = self._mueller_tc_prep(blm, mueller, lmax, kmax)
        
        # Decompose the discrete angle snapshots into Fourier coefficients
        tmp = self._pseudo_fft(tmp)

        # Truncate to save memory/compute for components with 0 power
        # kmax + 4 is necessary because Mueller mixing spreads power to m +/- 4
        truncated_comps = truncate_blm(tmp, self._lmax, self._kmax + 4)

        self._inter = []
        intertype = (
            ducc0.totalconvolve.Interpolator_f
            if single_precision
            else ducc0.totalconvolve.Interpolator
        )
        
        for cur_component in truncated_comps:
            if cur_component is None:
                # We can ignore this component entirely
                self._inter.append(None)
                continue

            self._inter.append(
                intertype(
                    sky=self._slm,
                    beam=cur_component.blms.astype(self._ctype, copy=False),
                    separate=False,
                    lmax=self._lmax,
                    kmax=cur_component.mmax,
                    npoints=npoints,
                    sigma_min=sigma_min,
                    sigma_max=sigma_max,
                    epsilon=epsilon,
                    nthreads=nthreads,
                )
            )

    def _mueller_tc_prep(self, blm: np.ndarray, mueller: np.ndarray, lmax: int, mmax: int) -> np.ndarray:
        """
        Internal: Combines the input beam with the Mueller matrix.
        """
        ncomp = blm.shape[0]
        C = mueller_to_C(mueller)

        # --- Step 1: Expand input blm to a dense grid including negative m ---
        # We need a range capable of holding mmax + 4 (due to Mueller mixing)
        # The internal dense buffer uses 0 as center.
        # buffer index = m + internal_mmax
        
        internal_mmax = mmax + 4
        dense_shape = (4, 2 * internal_mmax + 1, lmax + 1)
        blm2_dense = np.zeros(dense_shape, dtype=np.complex128)

        idx = 0
        for m in range(mmax + 1):
            sign = (-1) ** m
            lrange = slice(m, lmax + 1)
            idxrange = slice(idx, idx + lmax + 1 - m)
            
            # Map m to dense array index
            idx_pos = m + internal_mmax
            idx_neg = -m + internal_mmax

            # T component
            blm2_dense[0, idx_pos, lrange] = blm[0, idxrange]
            blm2_dense[0, idx_neg, lrange] = np.conj(blm[0, idxrange]) * sign
            
            # V component
            if ncomp > 3:
                blm2_dense[3, idx_pos, lrange] = blm[3, idxrange]
                blm2_dense[3, idx_neg, lrange] = np.conj(blm[3, idxrange]) * sign
            
            # E/B components
            if ncomp > 2:
                # positive m
                # Adri's notes [10] and [9]
                # spin +2
                p_plus = -blm[1, idxrange] - 1j * blm[2, idxrange]
                # spin -2
                p_minus = -blm[1, idxrange] + 1j * blm[2, idxrange]

                blm2_dense[1, idx_pos, lrange] = p_plus
                blm2_dense[2, idx_pos, lrange] = p_minus

                # negative m
                # Adri's notes [2]
                # spin +2
                blm2_dense[1, idx_neg, lrange] = np.conj(blm2_dense[2, idx_pos, lrange]) * sign
                # spin -2
                blm2_dense[2, idx_neg, lrange] = np.conj(blm2_dense[1, idx_pos, lrange]) * sign
                
            idx += lmax + 1 - m

        # --- Step 2: Compute mixing for 5 discrete angles ---
        sqrt2 = np.sqrt(2.0)
        nbeam = 5
        # The output needs to be flattened back to HEALPix-like format
        # but extended to mmax + 4
        res = np.zeros((nbeam, ncomp, nalm(lmax, mmax + 4)), dtype=self._ctype)

        for ibeam in range(nbeam):
            alpha = ibeam * np.pi / nbeam
            e2ia = np.exp(2j * alpha)
            e2iac = np.exp(-2j * alpha)
            e4ia = np.exp(4j * alpha)
            e4iac = np.exp(-4j * alpha)

            # Temporary dense buffer for the result of this angle
            blm_eff_dense = np.zeros_like(blm2_dense)

            # Loop over all m required for the output (from -(mmax+4) to +(mmax+4))
            # We iterate m from -internal_mmax to internal_mmax
            for m in range(-internal_mmax, internal_mmax + 1):
                lrange = slice(abs(m), lmax + 1)
                
                # Dense indices
                i_m = m + internal_mmax
                i_m_plus_2 = m + 2 + internal_mmax
                i_m_minus_2 = m - 2 + internal_mmax
                i_m_plus_4 = m + 4 + internal_mmax
                i_m_minus_4 = m - 4 + internal_mmax

                # Helper to safely get from dense array (returns 0 if out of bounds)
                # Since we padded blm2_dense to mmax+4, checking bounds for +/- 2 is safe
                # Checking +/- 4 might technically hit edge if m is at limit, but loop handles range.
                def get_d(comp, idx_m):
                    if idx_m < 0 or idx_m >= blm2_dense.shape[1]:
                        return 0j
                    return blm2_dense[comp, idx_m, lrange]

                # T component, Marta notes [4a]
                blm_eff_dense[0, i_m, lrange] = (
                      C[0, 0] * get_d(0, i_m)
                    + C[3, 0] * get_d(3, i_m)
                    + 1.0/sqrt2 * ((C[1, 0]*e2iac ) * get_d(1, i_m_minus_2)
                                  +(C[2, 0]*e2ia) * get_d(2, i_m_plus_2))
                )

                # E/B components, Marta notes [4b,c]
                blm_eff_dense[1, i_m, lrange] = (
                      (sqrt2*e2ia) * (C[0, 1] * get_d(0, i_m_plus_2)
                                     + C[3, 1] * get_d(3, i_m_plus_2))
                    + (C[2, 1]*e4ia) * get_d(2, i_m_plus_4)
                    +  C[1, 1]       * get_d(1, i_m)
                )
                blm_eff_dense[2, i_m, lrange] = (
                      (sqrt2*e2iac) * (C[0, 2] * get_d(0, i_m_minus_2)
                                    + C[3, 2] * get_d(3, i_m_minus_2))
                    + (C[1, 2]*e4iac) * get_d(1, i_m_minus_4)
                    +  C[2, 2]        * get_d(2, i_m)
                )

                # V component, Marta notes [4d]
                blm_eff_dense[3, i_m, lrange] = (
                      C[0, 3] * get_d(0, i_m)
                    + C[3, 3] * get_d(3, i_m)
                    + 1.0/sqrt2 * ((C[1, 3]*e2iac) * get_d(1, i_m_plus_2)
                                  +(C[2, 3]*e2ia)  * get_d(2, i_m_minus_2))
                )

            # --- Step 3: Convert back to original TEBV b_lm format ---
            idx_out = 0
            for m in range(internal_mmax + 1): # loop 0 to mmax+4
                lrange = slice(m, lmax + 1)
                idxrange = slice(idx_out, idx_out + lmax + 1 - m)
                i_m = m + internal_mmax # index in dense array

                # T component
                res[ibeam, 0, idxrange] = blm_eff_dense[0, i_m, lrange]
                
                # P/P* components -> back to E/B
                if ncomp > 2:
                    # Adri's notes [10]
                    # E = -0.5 * (P_plus + P_minus)
                    # B = 0.5j * (P_plus - P_minus)
                    p_plus = blm_eff_dense[1, i_m, lrange]
                    p_minus = blm_eff_dense[2, i_m, lrange]
                    
                    res[ibeam, 1, idxrange] = -0.5 * (p_plus + p_minus)
                    res[ibeam, 2, idxrange] = 0.5j * (p_plus - p_minus)

                # V component
                if ncomp > 3:
                    res[ibeam, 3, idxrange] = blm_eff_dense[3, i_m, lrange]
                
                idx_out += lmax + 1 - m

        return res

    # "Fourier transform" the blm at different alpha to obtain
    # blm(alpha) = out[0] + cos(2*alpha)*out[1] + sin(2*alpha)*out[2]
    #                     + cos(4*alpha)*out[3] + sin(4*alpha)*out[4]
    def _pseudo_fft(self, inp: np.ndarray) -> np.ndarray:
        out = np.zeros_like(inp)
        
        # Coefficients
        c1, s1 = np.cos(2*np.pi/5), np.sin(2*np.pi/5)
        c2, s2 = np.cos(4*np.pi/5), np.sin(4*np.pi/5)
        
        # a0
        out[0] = 0.2 * np.sum(inp, axis=0)
        
        # a1, a2 (2*alpha)
        out[1] = 0.4 * (inp[0] + c1*(inp[1]+inp[4]) + c2*(inp[2]+inp[3]))
        out[2] = 0.4 * (         s1*(inp[1]-inp[4]) + s2*(inp[2]-inp[3]))
        
        # a3, a4 (4*alpha)
        out[3] = 0.4 * (inp[0] + c2*(inp[1]+inp[4]) + c1*(inp[2]+inp[3]))
        out[4] = 0.4 * (         s2*(inp[1]-inp[4]) - s1*(inp[2]-inp[3]))
        
        return out

    def signal(self, *, ptg: np.ndarray, alpha: np.ndarray, strict_typing: bool):
        """Computes the convolved signal for a set of pointings and HWP angles.

        Parameters
        ----------
        ptg : numpy.ndarray((nptg, 3), dtype=numpy.float32/64)
            the input pointings in radians, in (theta, phi, psi) order
        alpha : numpy.ndarray((nptg,), dtype=numpy.float32/64)
            the HWP angles in radians
        strict_typing : bool
            if True, check that input types match the solver precision.

        Returns
        -------
        signal : numpy.ndarray((nptg,), dtype=numpy.float32/64)
            the signal measured by the detector
        """
        if strict_typing:
            if (ptg.dtype != self._ftype):
                raise TypeError(
                    "pointings are {} but they should be {}; consider "
                    "passing `strict_typing=False` to BeamConvolutionParameters.".format(
                        str(ptg.dtype), str(self._ftype),
                    )
                )
            if (alpha.dtype != self._ftype):
                raise TypeError(
                    "HWP angles are {} but they should be {}; consider "
                    "passing `strict_typing=False` to BeamConvolutionParameters.".format(
                        str(alpha.dtype), str(self._ftype),
                    )
                )

        ptg = ptg.astype(self._ftype, copy=False)
        alpha = alpha.astype(self._ftype, copy=False)
        
        # Initialize result with the DC term (component 0)
        if self._inter[0] is not None:
            res = self._inter[0].interpol(ptg)[0]
        else:
            res = np.zeros(ptg.shape[0], dtype=self._ftype)
            
        # Accumulate HWP modulated terms
        # Use in-place addition to save memory allocation
        if self._inter[1] is not None:
            res += np.cos(2*alpha) * self._inter[1].interpol(ptg)[0]
        if self._inter[2] is not None:
            res += np.sin(2*alpha) * self._inter[2].interpol(ptg)[0]
        if self._inter[3] is not None:
            res += np.cos(4*alpha) * self._inter[3].interpol(ptg)[0]
        if self._inter[4] is not None:
            res += np.sin(4*alpha) * self._inter[4].interpol(ptg)[0]
            
        return res
