# -*- coding: utf-8 -*-

"""A module for the 1D spectral extraction halftrace method"""

from bokeh.plotting import figure, show
from bokeh.models import Range1d
from bokeh.layouts import column
from hotsoss import locate_trace as lt
from hotsoss import plotting as plt
import numpy as np

from . import utilities as u


def extract(data, filt='CLEAR', radius=25, subarray='SUBSTRIP256', contam_end=688, **kwargs):
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
    contam_end: idx
        The first column where the upper half of
        order 1 is not contaminated by order 2

    Returns
    -------
    dict
        The wavelength array and time-series 1D counts and spectra
    """
    # Dictionary for results
    results = {}

    # Load the wavebins
    order1_wave, order2_wave = lt.trace_wavelengths(order=None, wavecal_file=None, npix=10, subarray=subarray)
    order1_bins, order2_bins = lt.wavelength_bins(subarray=subarray)
    coeffs = lt.trace_polynomial(subarray=subarray)

    # Reshape
    data, dims = u.to_3d(data)

    # NaN reference pixels
    data = u.nan_reference_pixels(data)

    # ========================================================================================
    # Order 1 trace (lower half of trace is uncontaminated) ==================================
    # ========================================================================================

    # Load the pixel halfmasks for order 1
    order1_lowermask, order1_uppermask = halfmasks(radius=radius, subarray=subarray, order=1)

    # Get the counts in the trace halves
    order1_lowercount = u.bin_counts(data, order1_bins, pixel_mask=order1_lowermask)
    order1_uppercount = u.bin_counts(data, order1_bins, pixel_mask=order1_uppermask)

    # Add lowercount and uppercount in the areas of no contamination
    order1_counts_uncontam = order1_lowercount[:, contam_end:] + order1_uppercount[:, contam_end:]
    order1_unc_uncontam = np.ones_like(order1_counts_uncontam)

    # ...double lowercount in the areas of upper trace contamination
    order1_counts_contam = order1_lowercount[:, :contam_end] * 2
    order1_unc_contam = np.ones_like(order1_counts_contam)

    # Construct order 1 counts from contaminated and uncontaminated pieces
    order1_counts = np.concatenate([order1_counts_contam, order1_counts_uncontam], axis=1)
    order1_unc = np.concatenate([order1_unc_contam, order1_unc_uncontam], axis=1)

    # Plot for visual inspection
    order1_plot = plt.plot_frame(data[0] * order1_lowermask, cols=[10, 500, 1500], title='Order 1 Signal', trace_coeffs=coeffs)

    results['order1'] = {'counts': order1_counts, 'unc': order1_unc, 'wavelength': order1_wave, 'filter': filt, 'subarray': subarray, 'plot': order1_plot}

    # ========================================================================================
    # Order 2 trace (upper half of trace is uncontaminated) ==================================
    # ========================================================================================

    if filt == 'CLEAR':

        # Load the pixel halfmasks for order 1
        order2_lowermask, order2_uppermask = halfmasks(radius=radius, subarray=subarray, order=2)

        # Get the counts in the trace halves
        order2_lowercount = u.bin_counts(data, order2_bins, pixel_mask=order2_lowermask)
        order2_uppercount = u.bin_counts(data, order2_bins, pixel_mask=order2_uppermask)

        # Add lowercount and uppercount in the areas of no contamination
        order2_counts_uncontam = order2_lowercount[:, contam_end:] + order2_uppercount[:, contam_end:]
        order2_unc_uncontam = np.ones_like(order2_counts_uncontam)

        # ...double lowercount in the areas of upper trace contamination
        order2_counts_contam = order2_uppercount[:, :contam_end] * 2
        order2_unc_contam = np.ones_like(order2_counts_contam)

        # Construct order 1 counts from contaminated and uncontaminated pieces
        order2_counts = np.concatenate([order2_counts_contam, order2_counts_uncontam], axis=1)
        order2_unc = np.concatenate([order2_unc_contam, order2_unc_uncontam], axis=1)

        # Plot for visual inspection
        order2_plot = plt.plot_frame(data[0] * order2_lowermask, cols=[10, 500, 1500], title='Order 1 Signal', trace_coeffs=coeffs)

        results['order2'] = {'counts': order2_counts, 'unc': order2_unc, 'wavelength': order2_wave, 'filter': filt, 'subarray': subarray, 'plot': order2_plot}

    return results


def halfmasks(subarray='SUBSTRIP256', radius=25, order=1, plot=False):
    """
    Generate a mask of the lower and upper halves of the first order trace

    Parameters
    ----------
    subarray: str
        The subarray to use, ['SUBSTRIP96', 'SUBSTRIP256', 'FULL']
    radius: int
        The radius in pixels of the trace
    plot: bool
        Plot the masks

    Returns
    -------
    lowermask, uppermask
        The lower and upper masks for the array
    """
    # Get the first order trace
    trace = lt.trace_polynomial(order=order)
    trace_vals = lt.trace_polynomial(order=order, evaluate=True)
    centers = np.ceil(trace_vals).astype(int)

    # Split into lower and upper halves
    dims = 2048, 256
    lowermask = np.zeros(dims)
    uppermask = np.zeros_like(lowermask)
    for n, center in enumerate(centers):
        if 0 < center - radius < 256:
            lowermask[n, center - radius:center] = 1
            uppermask[n, center + 1:center + 1 + radius] = 1

    # The trace cuts across each center pixel at an angle so calculate the fraction based on the area.
    # Get average between pixel centers to calculate the trace value at each pixel bounds.
    avg = [trace_vals[0]]
    for n in range(1, 2048):
        avg.append((trace_vals[n - 1] + trace_vals[n]) / 2.)
    avg.append(trace_vals[-1])

    # Calculate the fraction in the lower polygon
    rectangles = np.array([min(avg[n], avg[n+1]) - np.floor(avg[n]) for n in range(2048)])
    triangles = np.array([0.5 * (max(avg[n], avg[n + 1]) - min(avg[n], avg[n + 1])) for n in range(2048)])
    lower_frac = rectangles + triangles

    # Make a mask for the upper and lower halves of the trace with 0 in masked pixels,
    # 1 in unmasked pixels, and the calculated fraction in the center pixel
    for n, (center, frac) in enumerate(zip(centers, lower_frac)):
        if 0 < center < 256:
            lowermask[n, center] = frac
            uppermask[n, center] = 1. - frac

    # Reshape into subarray
    if subarray == 'SUBSTRIP96':
        dims = 2048, 96
        lowermask = lowermask[:, :96]
        uppermask = uppermask[:, :96]

    if subarray == 'FULL':
        dims = 2048, 2048
        full = np.zeros(dims)
        lowermask = np.pad(lowermask, ((0, 0), (1792, 0)))
        uppermask = np.pad(uppermask, ((0, 0), (1792, 0)))

    # Plot it
    if plot:

        source = dict(lower=[lowermask.T], upper=[uppermask.T], total=[lowermask.T + uppermask.T])
        tooltips = [("(x,y)", "($x{int}, $y{int})"), ("Lower", "@lower"), ("Upper", "@upper"), ('Total', '@total')]
        mask_plot = figure(x_range=Range1d(0, 2048), y_range=Range1d(0, dims[1]), width=900, height=int(dims[1]), tooltips=tooltips)
        mask_plot.image(source=source, image='total', x=0, y=0, dh=dims[1], dw=2048, alpha=0.1)
        show(mask_plot)

    return lowermask.T, uppermask.T
