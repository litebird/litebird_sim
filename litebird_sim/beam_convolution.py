import numpy as np
from .observations import Observation


def add_convolved_sky_to_observations(
    obs_list: list[Observation],
    slm_dictionary: dict[str, any],  # unconvolved sky a_lm
    blm_dictionary: dict[str, any],  # beam a_lm
    det2slm: dict[str, str],  # detector name -> slm name
    det2blm: dict[str, str],  # detector name -> blm name (could be identity)
    component: str = "tod",
):
    """Convolve sky maps with generic detector beams and add the resulting
    signal to TOD.

    Arguments
    ---------
    obs_list: list[Observation],
        List of Observation objects, containing detector names, pointings,
        and TOD data, to which the computed TOD are added.
    slm_dictionary:  dict[str, any]
        sky a_lm. Typically only one set of sky a_lm is needed per detector frequency
    blm_dictionary: dict[str, any]
        beam a_lm. Usually one set of a_lm is needed for every detector.
    det2slm: dict[str, str]
        converts detector name to a key for `slm_dictionary`
    det2slm: dict[str, str]
        converts detector name to a key for `blm_dictionary`
    component: str
        name of the TOD component to which the computed data shall be added
    """

    # These need to be provided by the user. They could be simply scalars, or
    # they could vary from detector to detector, not sure yet which is most useful.
    #    lmax, kmax = 1000, 5

    # find all involved detector names
    detnames = set()
    for obs in obs_list:
        for det in obs.name:
            detnames.add(det)

    for cur_det in detnames:
        # Set up the interpolator for this particular detector
        #        slm = slm_dictionary[det2slm[cur_det]]
        #        blm = blm_dictionary[det2blm[cur_det]]
        #        interp = ducc0.totalconvolve.Interpolator(sky=slm, beam=blm,
        #            separate=False, lmax=lmax, kmax=kmax, epsilon=1e-5, nthreads=1)

        # Now go through all the pointings for this detector
        # It might be advantageous to concatenate several chunks of observations
        # together - this can make interpolation a bit more efficient.
        # For now, let's just go through the chunks individually ...

        for cur_obs in obs_list:
            det_idx = list(cur_obs.name).index(cur_det)
            ptg = cur_obs.pointings[det_idx]
            psi = cur_obs.psi[det_idx]
            # Ducc requires pointing information as a single array with shape
            # (nptg, 3)
            ptgnew = np.empty((ptg.shape[0], 3), dtype=ptg.dtype)
            ptgnew[:, 0:2] = ptg
            ptgnew[:, 2] = psi

            # Get a reference to the TOD to which we should add our signal
        #    cur_tod = getattr(cur_obs, component)[det_idx]

        # Compute our signal and add it
        #    cur_tod += interp.interpol(ptgnew)
