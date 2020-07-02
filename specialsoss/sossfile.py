# -*- coding: utf-8 -*-

"""A module to handle NIRISS SOSS data in JWST pipeline file format"""

from copy import copy
from datetime import datetime, timedelta
import os
from pkg_resources import resource_filename

from astropy.io import fits
from astropy.time import Time
from hotsoss import locate_trace as lt
from hotsoss import plotting as plt
from hotsoss import utils
import numpy as np

from .utilities import to_3d


class SossFile:
    """
    A class object to handle JWST pipeline files for SOSS data
    """
    def __init__(self, filepath=None, **kwargs):
        """
        Initialize the SossFile object
        """
        # Calibration levels
        self.levels = ['uncal', 'ramp', 'rate', 'rateints', 'calints', 'x1dints']

        # Set default attributes to None
        self._file = None
        self.ext = None
        self.hdulist = None
        self.header = None
        self.data = None
        self.nints = None
        self.ngrps = None
        self.nrows = None
        self.ncols = None
        self.filter = None
        self.subarray = None
        self.median = None
        self.wavecal = None
        self.time = None
        self.star = None

        # Set the filepath to populate the attributes
        self.file = filepath

    def calibrate(self, configdir=None, outdir=None, **kwargs):
        """
        Pipeline process the file

        Parameters
        ----------
        configdir: str
            The directory containing the configuration files
        outdir: str
            The directory to put the calibrated files into
        """
        # Get config directory
        if configdir is None:
            configdir = resource_filename('specialsoss', 'files')

        # Get output directory
        if outdir is None:
            outdir = os.path.dirname(self.file)

        # Get basename
        basename = os.path.basename(self.file)
        file = os.path.join(outdir, basename)

        # Dict for new files
        new_files = {}

        if self.ext == 'uncal':

            # Create Detector1Pipeline instance
            cfg1_file = os.path.join(configdir, 'calwebb_tso1.cfg')
            from jwst.pipeline import Detector1Pipeline
            tso1 = Detector1Pipeline.call(self.file, save_results=True, config_file=cfg1_file, output_dir=outdir)

            # Calibrated files
            new_files['ramp'] = os.path.join(outdir, file.replace('_uncal.fits', '_ramp.fits'))
            new_files['rate'] = os.path.join(outdir, file.replace('_uncal.fits', '_rate.fits'))
            new_files['rateints'] = os.path.join(outdir, file.replace('_uncal.fits', '_rateints.fits'))

        if self.ext in ['rate', 'rateints']:

            # SPEC2 Pipeline
            cfg2_file = os.path.join(configdir, 'calwebb_tso-spec2.cfg')
            from jwst.pipeline import Spec2Pipeline
            tso2 = Spec2Pipeline(save_results=True, config_file=cfg2_file, output_dir=outdir)

            # Configure steps
            tso2.cube_build.skip = True
            tso2.extract_2d.skip = True
            tso2.bkg_subtract.skip = True
            tso2.msa_flagging.skip = True
            tso2.barshadow.skip = True
            tso2.extract_1d.save_results = True

            # Run the pipeline
            tso2.run(self.file)

            # Calibrated files
            new_files['calints'] = os.path.join(outdir, file.replace('_rateints.fits', '_calints.fits'))
            new_files['x1dints'] = os.path.join(outdir, file.replace('_rateints.fits', '_x1dints.fits'))

        else:

            print("Not sure how to calibrate a '{}' file right now.".format(self.ext))

        return new_files

    @property
    def file(self):
        return self._file

    @file.setter
    def file(self, filepath):
        """
        Setter for the file attribute

        Parameters
        ----------
        file: str
            The path to the file
        """
        if filepath is None:
            self._file = None

        elif isinstance(filepath, str):

            # Make sure the file exists
            if not os.path.exists(filepath):
                raise IOError("{} : Invalid file".format(filepath))

            # Determine processing level of input file
            if not any([filepath.endswith('_{}.fits'.format(level)) for level in self.levels]):
                raise ValueError('Not a recognized JWST file extension. Please use {}'.format(self.levels))

            # Save the filepath
            self._file = filepath

            # Save the filepath
            self.ext = filepath.split('_')[-1][:-5]

            # Load wavelength calibration
            self.load_wavecal()

            # Get the header and data from the FITS file
            hdulist = fits.open(filepath)
            self.hdulist = copy(hdulist)
            self.header = hdulist['PRIMARY'].header

            # If x1dints file, grab the extracted spectra
            if self.ext == 'x1dints':
                self.data = [hdulist[n].data['FLUX'] for n in range(len(hdulist)) if hdulist[n].name == 'EXTRACT1D']
                self.wavecal = [hdulist[n].data['WAVELENGTH'] for n in range(len(hdulist)) if hdulist[n].name == 'EXTRACT1D']

            # If 2D or more, save SCI extension as data
            else:
                self.data = hdulist['SCI'].data

            # Try to get 1D input used to make simulation (for extraction testing purposes)
            try:
                self.star = hdulist['STAR'].data
            except KeyError:
                pass

            # Close the file
            hdulist.close()

            # Observation parameters
            self.nints = self.header['NINTS']
            self.ngrps = self.header['NGROUPS']
            self.nframes = self.header['NFRAMES']
            self.nrows = self.header['SUBSIZE2']
            self.ncols = self.header['SUBSIZE1']
            self.filter = self.header['FILTER']
            self.frame_time = self.header['TFRAME']
            self.subarray = 'FULL' if self.nrows == 2048 else 'SUBSTRIP96' if self.nrows == 96 else 'SUBSTRIP256'

            # Determine the time axis given datetime and frame time
            time_str = '{} {}'.format(self.header['DATE-OBS'], self.header['TIME-OBS'])
            starttime = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S.%f")
            dt = timedelta(seconds=self.frame_time)
            self.time = Time(starttime + dt * np.arange(self.nframes))

            # Get time at end of each integration if 3D data
            if self.data.ndim == 3:
                self.time = self.time[self.ngrps-1::self.ngrps]

            # Otherwise time axis is irrelevant
            if self.data.ndim == 2:
                self.time = None

            # Compose a median image from the stack
            if self.ext != 'x1dints':
                if self.data.ndim == 4:
                    self.median = np.median(self.data, axis=(0, 1))
                elif self.data.ndim == 3:
                    self.median = np.median(self.data, axis=0)
                else:
                    self.median = self.data

        else:
            raise TypeError('File path must be a string. {} was given'.format(type(filepath)))

    def _get_frame(self, idx=None):
        """
        Retrieve some frame data

        Parameters
        ----------
        idx: int
            The index of the frame to retrieve
        """
        if isinstance(idx, int):

            # Reshape the data
            data, dims = to_3d(self.data)
            frame = data[idx]

        else:
            frame = self.median

        return frame

    def load_wavecal(self, file=None):
        """
        Load the wavelength calibration for orders 1, 2, and 3

        Parameters
        ----------
        file: str (optional)
            The path to a custom wavelength calibration file
        """
        # Load wavelength calibration file
        self.wavecal = utils.wave_solutions(subarray=self.subarray, file=file)

    def plot(self, scale='linear', coeffs=None, **kwargs):
        """
        Plot the frames of data

        Parameters
        ----------
        scale: str
            The scale to plot, ['linear', 'log']
        coeffs: sequence
            The polynomial coefficients of the traces
        """
        # Reshape the data
        data, dims = to_3d(self.data)

        # Make the figure
        title = '{} Frames'.format(self.ext)
        coeffs = lt.trace_polynomial()
        fig = plt.plot_frames(data, scale=scale, trace_coeffs=coeffs, wavecal=self.wavecal, title=title, **kwargs)

        return fig

    def __repr__(self):
        """
        Return the path to the file
        """
        return self.file or 'None'
