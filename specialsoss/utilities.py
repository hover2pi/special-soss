# -*- coding: utf-8 -*-

"""A module of shared tools for SOSS data"""

from copy import copy
import os
from pkg_resources import resource_filename

from bokeh.plotting import figure, show
from hotsoss.plotting import plot_frame
import numpy as np


def bin_counts(data, wavebins, pixel_mask=None, plot_bin=None):
    """
    Bin the counts in data given the wavelength bin information

    Parameters
    ----------
    data: array-like
        The 3D or 4D data to bin
    wavebins: sequence
        A list of lists of the pixels in each wavelength bin
    pixel_mask: array-like (optional)
        A 2D mask of 1s and 0s to apply to the data
    plot_bin: int
        The index of the bin to plot

    Returns
    -------
    np.ndarray
        The counts in each wavelength bin for each frame in data
    """
    # Reshape into 3D
    data, dims = to_3d(data)

    # Array to store counts
    counts = np.zeros((data.shape[0], len(wavebins)), dtype=float)

    # Apply the pixel mask by multiplying non-signal pixels by 0 before adding
    if isinstance(pixel_mask, np.ndarray) and pixel_mask.shape == data.shape[-2:]:
        data *= pixel_mask[None, :, :]

    # Add up the counts in each bin in each frame
    for n, (xpix, ypix) in enumerate(wavebins):
        cutoff = xpix < dims[-2]
        counts[:, n] = np.nansum(data[:, xpix[cutoff], ypix[cutoff]], axis=1)

    # Plot a bin for visual inspection
    if isinstance(plot_bin, int):

        # Make frame from binned pixels
        binpix = wavebins[plot_bin]
        binframe = copy(data[-1])
        binframe[binpix[0], binpix[1]] *= 10
        fig = plot_frame(binframe, title="Bin {}: {:.0f} counts in {:.2f} pixels".format(plot_bin, counts[-1][plot_bin], np.nansum(pixel_mask[binpix[0], binpix[1]])))
        show(fig)

    # Restore original shape
    counts = counts.reshape((*list(dims[:-2]), counts.shape[-1]))

    return counts


def combine_spectra(s1, s2):
    """
    Make a composite spectrum from two spectra

    Parameters
    ----------
    s1: sequence
        The [W, F, E] of the first spectrum
    s2: sequence
        The [W, F, E] of the second spectrum

    Returns
    -------
    sequence
        The [W, F, E] of the combined spectrum
    """
    # Remove NaN and zero wavelengths
    idx1 = np.where(s1[0] > 0.)[0]
    idx2 = np.where(s2[0] > 0.)[0]
    s1 = np.array([i[idx1] for i in s1])
    s2 = np.array([i[idx2] for i in s2])

    # Determine if overlapping
    overlap = True
    try:
        if s1[0][-1] > s1[0][0] > s2[0][-1] or s2[0][-1] > s2[0][0] > s1[0][-1]:
            overlap = False
    except IndexError:
        overlap = False

    # Concatenate and order two segments if no overlap
    if not overlap:

        # Drop uncertainties on both spectra if one is missing
        if s1[2] is None or s2[2] is None:
            s1 = s1[:2]
            s2 = s2[:2]

        # Concatenate arrays and sort by wavelength
        new_spec = np.concatenate([s1, s2], axis=1).T
        new_spec = new_spec[np.argsort(new_spec[:, 0])].T

    # Otherwise there are three segments, (left, overlap, right)
    else:

        # Get the left segemnt
        left = s1[:, s1[0] <= s2[0][0]]
        if not np.any(left):
            left = s2[:, s2[0] <= s1[0][0]]

        # Get the right segment
        right = s1[:, s1[0] >= s2[0][-1]]
        if not np.any(right):
            right = s2[:, s2[0] >= s1[0][-1]]

        # Get the overlapping segements
        o1 = s1[:, np.where((s1[0] < right[0][0]) & (s1[0] > left[0][-1]))].squeeze()
        o2 = s2[:, np.where((s2[0] < right[0][0]) & (s2[0] > left[0][-1]))].squeeze()

        # Get the resolutions
        r1 = s1.shape[1]/(max(s1[0])-min(s1[0]))
        r2 = s2.shape[1]/(max(s2[0])-min(s2[0]))

        # Make higher resolution s1
        if r1 < r2:
            o1, o2 = o2, o1

        # Interpolate s2 to s1
        o2_flux = np.interp(o1[0], s2[0], s2[1])

        # Get the average
        o_flux = np.nanmean([o1[1], o2_flux], axis=0)

        # Calculate uncertainties if possible
        if len(s2) == len(o1) == 3:
            o2_unc = np.interp(o1[0], s2[0], s2[2])
            o_unc = np.sqrt(o1[2]**2 + o2_unc**2)
            overlap = np.array([o1[0], o_flux, o_unc])
        else:
            overlap = np.array([o1[0], o_flux])
            left = left[:2]
            right = right[:2]

        # Make sure it is 2D
        if overlap.shape == (3,):
            overlap.shape = 3, 1
        if overlap.shape == (2,):
            overlap.shape = 2, 1

        # Concatenate the segments
        new_spec = np.concatenate([left, overlap, right], axis=1)

    return new_spec


def nan_reference_pixels(data):
    """
    Convert reference pixels in SOSS data to NaN values
    """
    # Convert to 3D
    data, dims = to_3d(data)

    # Left, right (all subarrays)
    if dims[-1] == 2048:
        data[:, :, :4] = np.nan
        data[:, :, -4:] = np.nan

    # Top (excluding SUBSTRIP96)
    if dims[-2] in [256, 2048]:
        data[:, -4:, :] = np.nan

    # Bottom (Only FULL frame)
    if dims[-2] == 2048:
        data[:, :4, :] = np.nan

    # Return to original shape
    data = data.reshape(dims)

    return data


def test_simulations():
    """
    Make test simulations using awesimsoss
    """
    try:
        from awesimsoss import BlackbodyTSO

        # Save location
        path = resource_filename('specialsoss', 'files/')

        # Delete old files
        os.system('rm {}/*.fits'.format(path))

        for filt in ['CLEAR', 'F277W']:
            for subarr in ['SUBSTRIP96', 'SUBSTRIP256']:

                tso = BlackbodyTSO(nints=2, ngrps=2, subarray=subarr, filter=filt)
                tso.export(os.path.join(path, '{}_{}_uncal.fits'.format(subarr, filt)))

    except ImportError:
        print("Please install awesimsoss to generate test simulations.")


def to_3d(data):
    """
    Class to convert arbitrary data into 3 dimensions

    Parameters
    ----------
    data: array-like
        The data to reshape

    Returns
    -------
    new_data, old_shape
        The reshaped data and the old shape
    """
    # Ensure it's an array
    if isinstance(data, list):
        data = np.asarray(data)

    # Get the data shape
    old_shape = data.shape

    # Convert to 3D
    if data.ndim == 4:
        new_data = data.reshape((old_shape[0]*old_shape[1], old_shape[2], old_shape[3]))
    elif data.ndim == 3:
        new_data = copy(data)
    elif data.ndim == 2:
        new_data = data[None, :, :]
    else:
        raise ValueError("{}: Data must be in 2, 3, or 4 dimensions.".format(old_shape))

    return new_data, old_shape
