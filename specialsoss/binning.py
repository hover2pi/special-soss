# -*- coding: utf-8 -*-

"""A module for the 1D spectral extraction binning method"""

from hotsoss import locate_trace as lt
import numpy as np

from . import utilities as u


def extract(data, filt, pixel_masks=None, subarray='SUBSTRIP256', **kwargs):
    """
    Extract the time-series 1D spectra from a data cube

    Parameters
    ----------
    data: array-like (optional)
        The CLEAR+GR700XD or F277W+GR700XD datacube
    filt: str
        The name of the filter, ['CLEAR', 'F277W']
    pixel_masks: sequence
        The pixel masks for each order
    subarray: str
        The subarray name

    Returns
    -------
    dict
        The wavelength array and time-series 1D counts and spectra
    """
    # Dictionary for results
    results = {}

    # Load the wavebins
    order1_wave, order2_wave = lt.trace_wavelengths(order=None, wavecal_file=None, npix=10, subarray='SUBSTRIP256')
    order1_bins, order2_bins = lt.wavelength_bins(subarray=subarray)

    # Reshape
    data, dims = u.to_3d(data)

    # NaN reference pixels
    data = u.nan_reference_pixels(data)

    # Load the pixel masks
    if pixel_masks is None:
        pixel_masks = np.ones((2, data.shape[-2], data.shape[-1]))

    # Bin the order 1 counts
    order1_counts = u.bin_counts(data, order1_bins, pixel_masks[0])
    order1_unc = np.ones_like(order1_counts)
    results['order1'] = {'counts': order1_counts, 'unc': order1_unc, 'wavelength': order1_wave, 'filter': filt, 'subarray': subarray}

    # Bin the order 2 counts
    order2_counts = u.bin_counts(data, order2_bins, pixel_masks[1])
    order2_unc = np.ones_like(order2_counts)
    results['order2'] = {'counts': order2_counts, 'unc': order2_unc, 'wavelength': order2_wave, 'filter': filt, 'subarray': subarray}

    return results
