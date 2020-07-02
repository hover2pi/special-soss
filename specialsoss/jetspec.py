# -*- coding: utf-8 -*-

"""A module for the 1D spectral extraction jetspec method"""

from pkg_resources import resource_filename

import astropy.units as q
from hotsoss import utils
from hotsoss import locate_trace as lt
import numpy as np
# import scipy.signal as sps


def extract(data, filt, subarray='SUBSTRIP256', units=q.erg/q.s/q.cm**2/q.AA, **kwargs):
    """
    Extract the time-series 1D spectra from a data cube

    Parameters
    ----------
    data: array-like
        The CLEAR+GR700XD or F277W+GR700XD datacube
    filt: str
        The name of the filter, ['CLEAR', 'F277W']
    subarray: str
        The subarray name
    units: astropy.units.quantity.Quantity
        The desired units for the output flux

    Returns
    -------
    dict
        The wavelength array and time-series 1D counts and spectra
    """
    # Get wavelengths
    wavelengths = lt.trace_wavelengths(order=None, wavecal_file=None, npix=10, subarray='SUBSTRIP256')

    # Calculate the mask
    masks = lt.order_masks(None, subarray=subarray)

    # TODO: Use error array
    err = np.zeros_like(data)

    # Run calcSpectrum
    wavelength, xpix, flux, stdspec, err, specbg = spectrum_extract(data, err)
    counts = np.zeros_like(flux)

    # Make results dictionary
    results = {'final': {'wavelength': wavelength[0], 'counts': counts, 'flux': flux[0], 'filter': filt, 'subarray': subarray}}

    return results


def spectrum_extract(data, err,\
                     jd=None,\
                     ncpu = 1,\
                     norders = 1,\
                     gain = 1.,\
                     v0 = 5.,\
                     spec_hw = [16,16,16],\
                     fitbghw = [33,28,20],\
                     expand = 1,\
                     bgdeg = 1,\
                     p3thresh = 5,\
                     p5thresh = 20,\
                     p7thresh = 20,\
                     fittype = 'meddata',\
                     window_len = 11,\
                     deg = 3,\
                     isplots = 1,\
                     tilt_correct = False,\
                     add_noise = False,\
                     return_flux = False,\
                     **kwargs):
    """
    Run spectral extraction routine on a file or directory of files

    Parameters
    ----------
    data: sequence
        The frames of data
    err: sequence
        The frames of errors
    jd: sequence
        The julian days of each frame
    ncpu
         Multiprocessing, code only generates figures when ncpu=1
    norders
         Number of spectral orders
    gain
         Gain
    v0
         Read noise
    spec_hw
         Half-width of spectral extraction along trace
    fitbghw
         Half-width of background along trace
    expand
         Increase resolution for tilt correction (not implemented)
    bgdeg
         Polynomial order for background subtraction
    p3thresh
         Reject outliers at X-sigma
    p5thresh
         Reject outliers at X-sigma
    p7thresh
         Reject outliers at X-sigma
    fittype
         Optimal weighting profiles: meddata, smooth, wavelet, wavelet2D, gauss, poly
    window_len
         Profile smoothing length
    deg
         Profile polynomial degree
    isplots
         Set from 0 to 7 for less/more figures
    tilt_correct
         Correct the tilt of the traces
    add_noise
         Add offsets, hot pixels, cosmic rays to data

    """
    # Reshape data
    if data.ndim == 4:
        data.shape = data.shape[0]*data.shape[1], data.shape[2], data.shape[3]
    if data.ndim == 2:
        data.shape = 1, data.shape[0], data.shape[1]

    # Re-order data and trim
    if data.shape[1] == 512:
        llim, ulim = 75, 331 # Trim the 512 array to 256 pixels
    else:
        llim, ulim = 0, -1
    data = data[:, llim:ulim, ::-1]

    # Create badpixel mask
    mask = np.ones(data.shape)

    # Define subarray regions for each spectral order
    ywindow = [[0,data.shape[1]],[0,data.shape[1]],[0,data.shape[1]]]
    xwindow = [[0,data.shape[2]],[520,data.shape[2]],[1130,data.shape[2]]]

    #Calculate rough trace and wavelength for each order
    meddata = np.mean(data,axis=0)
    #meddata = np.median(data,axis=0)
    #smmeddata = smoothing.smoothing(meddata, (3,3))
    trace = lt.trace_polynomial(evaluate=True)
    wave = lt.trace_wavelengths()
    xpix = [np.arange(start,stop,1) for start,stop in xwindow]

    # for n in range(norders):
    #     trace.append(calcTrace(meddata, mask[0], xwindow, order=n+1))
    #     wave.append(calcWave(np.arange(xwindow[n][1],xwindow[n][0],-1), order=n+1))
    #     if (trace[n].min() < spec_hw[n]) or (trace[n].min() < fitbghw[n]):
    #         print("WARNING: Spectral extraction goes out-of-bounds for given trace.")

    if add_noise:
        # Introduce amplifier offset into data
        data[:, :, 512 ] += 15
        data[:, :, 512:1024] += 3
        data[:, :, 1024:1536] += 7
        data[:, :, 1536:2048] += 2

        # Randomly add hot pixels
        nhot = 500
        hotval = 4*np.max(data)
        nints,ny,nx = data.shape
        randy = np.random.randint(0,ny,nhot)
        randx = np.random.randint(0,nx,nhot)
        data[:,randy,randx] = hotval

        mask2 = np.copy(mask)
        mask2[:,randy,randx] = 0

        # Insert cosmic rays
        randy = np.random.randint(10,ny-110,nints)
        randx = np.random.randint(10,nx-110,nints)
        randlen = np.random.randint(50,100,nints)
        for i in range(nints):
            data[i,randy[i],randx[i]:randx[i]+randlen[i]] = hotval
    else:

        nints, ny, nx = data.shape

    # Tilt correction
    if tilt_correct:
        print("Calculating slit shift values...")
        slitshift, shift_values, yfit = calc_slitshift2(data[0], ev.xrange, ywindow, xwindow)
    else:
        # No tilt correction
        slitshift = np.zeros(data.shape[2])

    # Initialize arrays
    global spectra, stdspe, specerr, specbg
    spectra = []
    stdspec = []
    specerr = []
    specbg = []
    for n in range(norders):
        spectra.append(np.zeros((nints, xwindow[n][1]-xwindow[n][0])))
        stdspec.append(np.zeros((nints, xwindow[n][1]-xwindow[n][0])))
        specerr.append(np.zeros((nints, xwindow[n][1]-xwindow[n][0])))
        specbg.append(np.zeros((nints, xwindow[n][1]-xwindow[n][0])))

    # Store data in arrays
    def store(arg):
        '''
        Write spectra, uncertainties, and background to arrays
        '''
        spectrum, standardspec, specstd, stdbg, m = arg
        for n in range(norders):
            spectra[n][m] = spectrum[n]
            stdspec[n][m] = standardspec[n]
            specerr[n][m] = specstd[n]
            specbg[n][m] = stdbg[n]
        return

    print('Processing data frames...')
    print('Total # of integrations: '+str(nints))

    if ncpu == 1:
        # Only 1 CPU
        for m in range(nints):
            store(calcSpectrum(data[m], mask[m], slitshift, \
                                      xwindow, ywindow, trace, gain, \
                                      v0, spec_hw, fitbghw, m, \
                                      p3thresh=p3thresh, p5thresh=p5thresh, \
                                      p7thresh=p7thresh, meddata=meddata, \
                                      fittype=fittype, window_len=window_len, \
                                      deg=deg, expand=expand, isplots=isplots, \
                                      bgdeg=1))

    else:
        # Multiple CPUs
        pool = mp.Pool(ncpu)
        for m in range(nints):
            res = pool.apply_async(calcSpectrum, \
                                   args=(data[m], mask[m], \
                                         slitshift, xwindow, \
                                         ywindow, trace, gain, \
                                         v0, spec_hw, fitbghw, m, \
                                         p3thresh, p5thresh, p7thresh, \
                                         meddata, fittype, window_len, \
                                         deg, expand, False, \
                                         bgdeg), callback=store)
        pool.close()
        pool.join()
        res.wait()

    # Symmetrize the data so it can be contained in arrays
    wave, xpix, spectra, stdspec, specerr, specbg = [symmetrize(arr) \
                for arr in [wave, xpix, spectra, stdspec, specerr, specbg]]

   # Convert to flux
    if return_flux:
        for i,spec in enumerate(spectra):

            # Get the wavelength dependent relative scaling for the order
            scaling = ADUtoFlux(i+1)
            response = np.interp(wave[i], *scaling, left=np.nan, right=np.nan)

            # Convert each spectrum from ADU/s to erg/s/cm2
            for j,s in enumerate(spec):
                spectra[i][j] *= response

    # Return or write to FITS file
    return wave, xpix, spectra, stdspec, specerr, specbg


def calcSpectrum(data, mask, slitshift, xwindow, ywindow, trace, gain, v0, \
                 spec_hw, fitbghw, m, p3thresh=5, p5thresh=10, p7thresh=10, \
                 meddata=None, fittype='smooth', window_len=150, deg=3, \
                 expand=1, bgdeg=1, isplots=1):
    '''
    Driver routine for optimal spectral extraction

    Parameters
    ----------
    data: array-like
        The 2D+ data that contains the spectrum images
    mask: array-like
        The mask generated by calcTrace()
    slitshift: array-like
        ?
    xwindow: sequence
        The upper and lower limits of pixels in the x direction
    ywindow: sequence
        The upper and lower limits of pixels in the y direction
    trace:
    gain: float
        The gain
    v0: float
        The read noise
    spec_hw: sequence
        The half-width of the spectral extraction along each order trace
    fitbghw: sequence
        The half-width of the background along each order trace
    m: int
        The number of the integration in the exposure
    p3thresh: float
        Reject outliers at X-sigma
    p5thresh: float
        Reject outliers at X-sigma
    p7thresh: float
        Reject outliers at X-sigma
    deg: int
        Degree of polynomial fit of profile

    Returns
    -------
    spectrum, stdspec, specstd, stdbg, m: sequence
        A sequence of output arrays

    History
    -------
    Written by Kevin Stevenson      September 2016
    Updated by Joe Filippazzo       February 2020
    '''
    norders = len(trace)

    #Generate background mask
    #Ones indicate pixels to be used for background estimate
    bgmask = np.ones(data.shape)
    for n in range(norders):
        # Set limits on the background
        y1 = (trace[n] - fitbghw[n]*expand).astype(int)
        y2 = (trace[n] + fitbghw[n]*expand).astype(int)
        for i in range(xwindow[n][0],xwindow[n][1]):
            bgmask[y1[i-xwindow[n][0]]:y2[i-xwindow[n][0]], i] = 0

    subny = []
    subnx = []
    stdspec = []
    stdvar = []
    stdbg = []
    spectrum = []
    specstd = []
    for n in range(norders):
        #Select subarray region
        subny.append(ywindow[n][1] - ywindow[n][0])
        subnx.append(xwindow[n][1] - xwindow[n][0])

        subdata = data[ywindow[n][0]:ywindow[n][1], xwindow[n][0]:xwindow[n][1]]
        #suberr = err[ywindow[n][0]:ywindow[n][1],xwindow[n][0]:xwindow[n][1]]
        submask = mask[ywindow[n][0]:ywindow[n][1], xwindow[n][0]:xwindow[n][1]]
        subbgmask = bgmask[ywindow[n][0]:ywindow[n][1], xwindow[n][0]:xwindow[n][1]]
        submeddata = meddata[ywindow[n][0]:ywindow[n][1], xwindow[n][0]:xwindow[n][1]]

        cordata = subdata
        cormask = submask
        corbgmask = subbgmask
        cormeddata = submeddata

        # STEP 3: Fit sky background with out-of-spectra data
        corbg = np.zeros((cordata.shape))
        # corbg, cormask = fitbg2(cordata, cormask, corbgmask, deg=bgdeg, threshold=p3thresh, isrotate=2, isplots=isplots)
        subdata = cordata
        submask = cormask
        bg = corbg

        # STEP 2: Calculate variance
        bgerr = np.std(bg, axis=1)/np.sqrt(np.sum(submask, axis=1))
        bgerr[np.where(np.isnan(bgerr))] = 0.
        v0 += np.mean(bgerr**2)
        variance = abs(subdata) / gain + v0
        # Perform background subtraction
        subdata -= bg

        # Shift spectrum along trace for nth spectral order
        y1 = (trace[n] - spec_hw[n]).astype(int)
        y2 = (trace[n] + spec_hw[n]).astype(int)
        trdata = np.zeros((spec_hw[n]*2,subnx[n]))
        trmask = np.zeros((spec_hw[n]*2,subnx[n]),dtype=bool)
        trvar = np.zeros((spec_hw[n]*2,subnx[n]))
        trbg = np.zeros((spec_hw[n]*2,subnx[n]))
        trmeddata = np.zeros((spec_hw[n]*2,subnx[n]))
        for i in range(subnx[n]):
            # Occasionally there is a broadcasting problem here. Why?
            try:
                trdata[:,i] = subdata[y1[i]:y2[i],i]
                trmask[:,i] = submask[y1[i]:y2[i],i]
                trvar[:,i] = variance[y1[i]:y2[i],i]
                trbg[:,i] = bg[y1[i]:y2[i],i]
                trmeddata[:,i] = submeddata[y1[i]:y2[i],i]
            except ValueError:
                pass

        # STEP 4: Extract standard spectrum and its variance
        stdspec.append(np.sum(trdata*trmask, axis=0))
        stdvar.append(np.sum(trvar*trmask, axis=0))
        stdbg.append(np.sum(trbg*trmask, axis=0))

        # Extract optimal spectrum with uncertainties
        spectrum.append(np.zeros((stdspec[-1].shape)))
        specstd.append(np.zeros((stdspec[-1].shape)))
        #FINDME: Do I really need to reshape data if I'm using median data to weight optimal?
        spectrum[n], specstd[n], tempmask = optimize(trdata, trmask, trbg, stdspec[-1], gain, v0, p5thresh=p5thresh, p7thresh=p7thresh, fittype=fittype, window_len=window_len, deg=deg, n=m, meddata=trmeddata)

        spectrum[n][np.where(np.isnan(spectrum[n]))[0]] = 0
        specstd [n][np.where(np.isnan(specstd[n]))[0]] = np.inf
        #np.sqrt(stdvar[np.where(np.isnan(specstd))[0]])
        trmask = tempmask

    return spectrum, stdspec, specstd, stdbg, m


def fitbg(dataim, mask, x1, x2, deg=1, threshold=5, isrotate=False, isplots=False):
    '''
    Fit sky background with out-of-spectra data

    HISTORY
    -------
    Written by Kevin Stevenson
    Removed [::-1] for LDSS3                May 2013
    Modified x1 and x2 to allow for arrays  Feb 2014
    '''

    # Assume x is the spatial direction and y is the wavelength direction
    # Otherwise, rotate array
    if isrotate == 1:
        dataim = dataim[::-1].T
        mask = mask[::-1].T
    elif isrotate == 2:
        dataim = dataim.T
        mask = mask.T

    #Convert x1 and x2 to array, if need be
    ny, nx = np.shape(dataim)
    if type(x1) == int or type(x1) == np.int64:
        x1 = np.zeros(ny,dtype=int)+x1
    if type(x2) == int or type(x2) == np.int64:
        x2 = np.zeros(ny,dtype=int)+x2

    if deg < 0:
        # Calculate median background of entire frame
        # Assumes all x1 and x2 values are the same
        submask = np.concatenate((  mask[:,
        x1[0]].T,  mask[:,x2[0]+1:nx].T)).T
        subdata = np.concatenate((dataim[:,
        x1[0]].T,dataim[:,x2[0]+1:nx].T)).T
        bg = np.zeros((ny,nx)) + np.median(subdata[np.where(submask)])
    elif deg == None:
        
        # No background subtraction
        bg = np.zeros((ny,nx))
    else:
        degs = np.ones(ny)*deg
        #degs[np.where(np.sum(mask[:, x1],axis=1) < deg)] = 0
        #degs[np.where(np.sum(mask[:,x2+1:nx],axis=1) < deg)] = 0
        # Initiate background image with zeros
        bg = np.zeros((ny,nx))
        # Fit polynomial to each column
        for j in range(ny):
            nobadpixels = False
            # Create x indices for background sections of frame
            xvals = np.concatenate((range(x1[j]), range(x2[j]+1,nx))).astype(int)
            # If too few good pixels then average
            if (np.sum(mask[j,:x1[j]]) < deg) or (np.sum(mask[j,x2[j]+1:nx]) < deg):
                degs[j] = 0
            while (nobadpixels == False):
                try:
                    goodxvals = xvals[np.where(mask[j,xvals])]
                except:
                    print(j)
                    print(xvals)
                    print(np.where(mask[j,xvals]))
                    return
                dataslice = dataim[j,goodxvals]
                # Check for at least 1 good x value
                if len(goodxvals) == 0:
                    #print(j,ny)
                    nobadpixels = True      #exit while loop
                    #Use coefficients from previous row
                else:
                    # Fit along spatial direction with a polynomial of degree 'deg'
                    coeffs = np.polyfit(goodxvals, dataslice, deg=degs[j])
                    # Evaluate model at goodexvals
                    model = np.polyval(coeffs, goodxvals)
                    #model = smooth.smooth(dataslice, window_len=window_len, window=windowtype)
                    #model = sps.medfilt(dataslice, window_len)

                    # Calculate residuals and number of sigma from the model
                    residuals = dataslice - model
                    stdres = np.std(residuals)
                    if stdres == 0:
                        stdres = np.inf
                    stdevs = np.abs(residuals) / stdres
                    # Find worst data point
                    loc = np.argmax(stdevs)
                    # Mask data point if > threshold
                    if stdevs[loc] > threshold:
                        mask[j,goodxvals[loc]] = 0
                    else:
                        nobadpixels = True      #exit while loop

            # Evaluate background model at all points, write model to background image
            if len(goodxvals) != 0:
                bg[j] = np.polyval(coeffs, range(nx))
                #bg[j] = np.interp(range(nx), goodxvals, model)

    if isrotate == 1:
        bg = (bg.T)[::-1]
        mask = (mask.T)[::-1]
    elif isrotate == 2:
        bg = (bg.T)
        mask = (mask.T)

    return bg, mask #,variance


def fitbg2(dataim, mask, bgmask, deg=1, threshold=5, isrotate=False, isplots=False):
    '''
    Fit sky background with out-of-spectra data

    HISTORY
    -------
    Written by Kevin Stevenson      September 2016
    '''

    # Assume x is the spatial direction and y is the wavelength direction
    # Otherwise, rotate array
    if isrotate == 1:
        dataim = dataim[::-1].T
        mask = mask[::-1].T
        bgmask = bgmask[::-1].T
    elif isrotate == 2:
        dataim = dataim.T
        mask = mask.T
        bgmask = bgmask.T

    # Initiate background image with zeros
    ny, nx = np.shape(dataim)
    bg = np.zeros((ny,nx))
    mask2 = mask*bgmask
    if deg < 0:
        # Calculate median background of entire frame
        bg  += np.median(data[np.where(mask2)])
    elif deg == None:
        
        # No background subtraction
        pass
    else:
        degs = np.ones(ny)*deg
        # Fit polynomial to each column
        for j in range(ny):
            nobadpixels = False
            # Create x indices for background sections of frame
            xvals = np.where(bgmask[j] == 1)[0]
            # If too few good pixels on either half of detector then compute average
            if True:#(np.sum(mask2[j,:nx/2]) < deg) or (np.sum(mask2[j,nx/2:nx]) < deg):
                degs[j] = 0
            while (nobadpixels == False):
                try:
                    goodxvals = xvals[np.where(mask[j,xvals])]
                except:
                    print(j)
                    print(xvals)
                    print(np.where(mask[j,xvals]))
                    return
                dataslice = dataim[j,goodxvals]
                # Check for at least 1 good x value
                if len(goodxvals) == 0:
                    #print(j,ny)
                    nobadpixels = True      #exit while loop
                    #Use coefficients from previous row
                else:
                    # Fit along spatial direction with a polynomial of degree 'deg'
                    coeffs = np.polyfit(goodxvals, dataslice, deg=degs[j])
                    # Evaluate model at goodexvals
                    model = np.polyval(coeffs, goodxvals)
                    #model = smooth.smooth(dataslice, window_len=window_len, window=windowtype)
                    #model = sps.medfilt(dataslice, window_len)

                    # Calculate residuals
                    residuals = dataslice - model
                    # Find worst data point
                    loc = np.argmax(np.abs(residuals))
                    # Calculate standard deviation of points excluding worst point
                    ind = list(range(len(residuals)))
                    ind.remove(loc)
                    stdres = np.std(residuals[ind])
                    if stdres == 0:
                        stdres = np.inf
                    # Calculate number of sigma from the model
                    stdevs = np.abs(residuals) / stdres
                    # Mask data point if > threshold
                    if stdevs[loc] > threshold:
                        mask[j,goodxvals[loc]] = 0
                    else:
                        nobadpixels = True      #exit while loop

            # Evaluate background model at all points, write model to background image
            if len(goodxvals) != 0:
                bg[j] = np.polyval(coeffs, range(nx))
                #bg[j] = np.interp(range(nx), goodxvals, model)

    if isrotate == 1:
        bg = (bg.T)[::-1]
        mask = (mask.T)[::-1]
        bgmask = (bgmask.T)[::-1]
    elif isrotate == 2:
        bg = (bg.T)
        mask = (mask.T)
        bgmask = (bgmask.T)

    return bg, mask #,variance


def profile_poly(subdata, mask, deg=3, threshold=10, isplots=False):
    '''
    Construct normalized spatial profile using polynomial fits along the wavelength direction
    '''
    submask = np.copy(mask)
    ny, nx = np.shape(subdata)
    profile = np.zeros((ny, nx))
    maxiter = nx
    for j in range(ny):
        nobadpixels = False
        iternum = 0
        while (nobadpixels == False) and (iternum < maxiter):
            dataslice = np.copy(subdata[j])     #Do not want to alter original data
            # Replace masked points with median of nearby points
            for ind in np.where(submask[j] == 0)[0]:
                dataslice[ind] = np.median(dataslice[np.max((0,ind-10)):ind+11])

            # Smooth each row
            coeffs = np.polyfit(range(nx), dataslice, deg)
            model = np.polyval(coeffs, range(nx))

            # Calculate residuals and number of sigma from the model
            residuals = submask[j]*(dataslice - model)
            stdevs = np.abs(residuals) / np.std(residuals)
            # Find worst data point
            loc = np.argmax(stdevs)
            # Mask data point if > threshold
            if stdevs[loc] > threshold:
                nobadpixels = False
                submask[j,loc] = 0
            else:
                nobadpixels = True      #exit while loop
            iternum += 1

        profile[j] = model
        if iternum == maxiter:
            print('WARNING: Max number of iterations reached for dataslice ' + str(j))

    # Enforce positivity
    profile[np.where(profile < 0)] = 0

    # Normalize along spatial direction
    profile /= np.sum(profile, axis=0)

    return profile


def profile_smooth(subdata, mask, threshold=10, window_len=21, windowtype='hanning', isplots=False):
    '''
    Construct normalized spatial profile using a smoothing function
    '''
    submask = np.copy(mask)
    ny, nx = np.shape(subdata)
    profile = np.zeros((ny, nx))
    maxiter = nx
    for j in range(ny):
        # Check for good pixels in row
        if np.sum(submask[j]) > 0:
            nobadpixels = False
            iternum = 0
            maxiter = np.sum(submask[j])
            while (nobadpixels == False) and (iternum < maxiter):
                dataslice = np.copy(subdata[j])     #Do not want to alter original data
                # Replace masked points with median of nearby points
                #dataslice[np.where(submask[j] == 0)] = 0
                #FINDME: Code below appears to be effective, but is slow for lots of masked points
                for ind in np.where(submask[j] == 0)[0]:
                    dataslice[ind] = np.median(dataslice[np.max((0,ind-10)):ind+11])

                # Smooth each row
                #model = smooth.smooth(dataslice, window_len=window_len, window=windowtype)
                model = smooth.medfilt(dataslice, window_len)

                # Calculate residuals and number of sigma from the model
                igoodmask = np.where(submask[j] == 1)[0]
                residuals = submask[j]*(dataslice - model)
                stdevs = np.abs(residuals[igoodmask]) / np.std(residuals[igoodmask])
                # Find worst data point
                loc = np.argmax(stdevs)
                # Mask data point if > threshold
                if stdevs[loc] > threshold:
                    nobadpixels = False
                    submask[j,igoodmask[loc]] = 0
                else:
                    nobadpixels = True      #exit while loop
                iternum += 1
            # Copy model slice to profile
            profile[j] = model
            if iternum == maxiter:
                print('WARNING: Max number of iterations reached for dataslice ' + str(j))

    # Enforce positivity
    profile[np.where(profile < 0)] = 0

    # Normalize along spatial direction
    profile /= np.sum(profile, axis=0)

    return profile


def profile_meddata(data, mask, meddata, threshold=10, isplots=0):
    '''
    Construct normalized spatial profile using median of all data frames
    '''
    profile = np.copy(meddata*mask)
    # Enforce positivity
    profile[np.where(profile < 0)] = 0
    # Normalize along spatial direction
    profile /= np.sum(profile, axis=0)

    return profile


def profile_wavelet(subdata, mask, wavelet, numlvls, isplots=False):
    '''
    Construct normalized spatial profile using wavelets. This function performs 1D image denoising using BayesShrink soft thresholding.
    Ref: Chang et al. "Adaptive Wavelet Thresholding for Image Denoising and Compression", 2000
    '''
    import pywt
    submask = np.copy(mask)
    ny, nx = np.shape(subdata)
    profile = np.zeros((ny, nx))

    for j in range(ny):
        #Perform wavelet decomposition
        dec = pywt.wavedec(subdata[j],wavelet)
        #Estimate noise variance
        noisevar = np.inf
        for i in range(-1,-numlvls-1,-1):
            noisevar = np.min([(np.median(np.abs(dec[i]))/0.6745)**2,noisevar])
        #At each level of decomposition...
        for i in range(-1,-numlvls-1,-1):
            #Estimate variance at level i then compute the threshold value
            sigmay2 = np.mean(dec[i]*dec[i])
            sigmax = np.sqrt(np.max([sigmay2-noisevar,0]))
            threshold = np.max(np.abs(dec[i]))
            #if sigmax == 0 or i == -1:
            #    threshold = np.max(np.abs(dec[i]))
            #else:
            #    threshold = noisevar/sigmax
            #Compute less noisy coefficients by applying soft thresholding
            dec[i] = map (lambda x: pywt.thresholding.soft(x,threshold), dec[i])

        profile[j] = pywt.waverec(dec,wavelet)[:nx]

    # Enforce positivity
    profile[np.where(profile < 0)] = 0
    # Normalize along spatial direction
    profile /= np.sum(profile, axis=0)

    return profile


def profile_wavelet2D(subdata, mask, wavelet, numlvls, isplots=False):
    '''
    Construct normalized spatial profile using wavelets. This function performs 2D image denoising using BayesShrink soft thresholding.
    Ref: Chang et al. "Adaptive Wavelet Thresholding for Image Denoising and Compression", 2000
    '''
    import pywt
    submask = np.copy(mask)
    ny, nx = np.shape(subdata)
    profile = np.zeros((ny, nx))

    #Perform wavelet decomposition
    dec = pywt.wavedec2(subdata,wavelet)
    #Estimate noise variance
    noisevar = np.inf
    for i in range(-1,-numlvls-1,-1):
        noisevar = np.min([(np.median(np.abs(dec[i]))/0.6745)**2,noisevar])
    #At each level of decomposition...
    for i in range(-1,-numlvls-1,-1):
        #Estimate variance at level i then compute the threshold value
        sigmay2 = np.mean((dec[i][0]*dec[i][0]+dec[i][1]*dec[i][1]+dec[i][2]*dec[i][2])/3.)
        sigmax = np.sqrt(np.max([sigmay2-noisevar,0]))
        threshold = np.max(np.abs(dec[i]))
        #if sigmax == 0:
        #    threshold = np.max(np.abs(dec[i]))
        #else:
        #    threshold = noisevar/sigmax
        #Compute less noisy coefficients by applying soft thresholding
        dec[i] = map (lambda x: pywt.thresholding.soft(x,threshold), dec[i])

    profile = pywt.waverec2(dec,wavelet)[:ny,:nx]

    # Enforce positivity
    profile[np.where(profile < 0)] = 0
    # Normalize along spatial direction
    profile /= np.sum(profile, axis=0)

    return profile


def profile_gauss(subdata, mask, threshold=10, guess=None, isplots=False):
    '''
    Construct normalized spatial profile using a Gaussian smoothing function
    '''
    submask = np.copy(mask)
    ny, nx = np.shape(subdata)
    profile = np.zeros((ny, nx))
    maxiter = ny
    for i in range(nx):
        nobadpixels = False
        iternum = 0
        dataslice = np.copy(subdata[:,i])     #Do not want to alter original data

        # Set initial guess if none given
        guess = [ny/10.,np.argmax(dataslice),dataslice.max()]
        while (nobadpixels == False) and (iternum < maxiter):
            #if guess == None:
                #guess = g.old_gaussianguess(dataslice, np.arange(ny), mask=submask[:,i])
            # Fit Gaussian to each column
            if sum(submask[:,i]) >= 3:
                params, err = g.fitgaussian(dataslice, np.arange(ny), mask=submask[:,i], fitbg=0, guess=guess)
            else:
                params = guess
                err = None
            # Create model
            model = g.gaussian(np.arange(ny), params[0], params[1], params[2])

            # Calculate residuals and number of sigma from the model
            residuals = submask[:,i]*(dataslice - model)
            if np.std(residuals) == 0:
                stdevs = np.zeros(residuals.shape)
            else:
                stdevs = np.abs(residuals) / np.std(residuals)

            # Find worst data point
            loc = np.argmax(stdevs)

            # Mask data point if > threshold
            if stdevs[loc] > threshold:

                # Check for bad fit, possibly due to a bad pixel
                if i > 0 and (err == None or abs(params[0]) < abs(0.2*guess[0])):

                    # Remove brightest pixel within region of fit
                    loc = params[1]-3 + np.argmax(dataslice[params[1]-3:params[1]+4])

                else:
                    guess = abs(params)
                submask[loc,i] = 0
            else:
                nobadpixels = True      #exit while loop
                guess = abs(params)
            iternum += 1

        profile[:,i] = model
        if iternum == maxiter:
            print('WARNING: Max number of iterations reached for dataslice ' + str(i))

    # Enforce positivity
    profile[np.where(profile < 0)] = 0

    # Normalize along spatial direction
    profile /= np.sum(profile, axis=0)

    return profile


def optimize(subdata, mask, bg, spectrum, Q, v0, p5thresh=10, p7thresh=10, fittype='smooth', window_len=21, deg=3, windowtype='hanning', n=0, isplots=False, meddata=None):
    '''
    Extract optimal spectrum with uncertainties
    '''
    submask = np.copy(mask)
    ny, nx = subdata.shape
    isnewprofile = True

    # Loop through steps 5-8 until no more bad pixels are uncovered
    while(isnewprofile == True):
        # STEP 5: Construct normalized spatial profile
        if fittype == 'smooth':
            profile = profile_smooth(subdata, submask, threshold=p5thresh, window_len=window_len, windowtype=windowtype, isplots=isplots)
        elif fittype == 'meddata':
            profile = profile_meddata(subdata, submask, meddata, threshold=p5thresh, isplots=isplots)
        elif fittype == 'wavelet2D':
            profile = profile_wavelet2D(subdata, submask, wavelet='bior5.5', numlvls=3, isplots=isplots)
        elif fittype == 'wavelet':
            profile = profile_wavelet(subdata, submask, wavelet='bior5.5', numlvls=3, isplots=isplots)
        elif fittype == 'gauss':
            profile = profile_gauss(subdata, submask, threshold=p5thresh, guess=None, isplots=isplots)
        else:
            profile = profile_poly(subdata, submask, deg=deg, threshold=p5thresh)

        isnewprofile = False
        isoutliers = True
        # Loop through steps 6-8 until no more bad pixels are uncovered
        while(isoutliers == True):
            # STEP 6: Revise variance estimates
            expected = profile*spectrum
            variance = np.abs(expected + bg) / Q + v0

            # STEP 7: Mask cosmic ray hits
            stdevs = np.abs(subdata - expected)*submask / np.sqrt(variance)
            isoutliers = False
            if len(stdevs) > 0:
                # Find worst data point in each column
                loc = np.argmax(stdevs, axis=0)
                # Mask data point if std is > p7thresh
                for i in range(nx):
                    if stdevs[loc[i],i] > p7thresh:
                        isnewprofile = True
                        isoutliers = True
                        submask[loc[i],i] = 0

                        # Check for insufficient number of good points
                        if sum(submask[:,i]) < ny/2.:
                            submask[:,i] = 0

            # STEP 8: Extract optimal spectrum
            denom = np.sum(profile*profile*submask/variance, axis=0)
            denom[np.where(denom == 0)] = np.inf
            spectrum = np.sum(profile*submask*subdata/variance, axis=0) / denom

    # Calculate variance of optimal spectrum
    specvar = np.sum(profile*submask, axis=0) / denom

    # Return spectrum and uncertainties
    return spectrum, np.sqrt(specvar), submask

def symmetrize(data):
    '''
    Make the list of mismatched array sizes into a single array 
    by padding smaller arrays with NaNs
    
    Parameters
    ----------
    data: list
        A list of arrays
    
    Returns
    -------
    arr: array-like
        A symmetric array containing the input data
    '''
    # Get the largest array shape
    max_shape = max([i.shape for i in data], key=lambda x: x[-1])
    shape = [len(data)]+list(max_shape)

    # Make an array of nans
    arr = np.zeros(shape)*np.nan

    # Paste the data into the new array
    for i,l1 in enumerate(data):
        if len(shape)==3:
            for j,l2 in enumerate(l1):
                arr[i,j,:len(l2)] = l2
        elif len(shape)==2:
            arr[i,:len(l1)] = l1

    return arr
