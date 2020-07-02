#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `utilities` module."""

import unittest

from hotsoss import locate_trace as lt
import numpy as np

from specialsoss import utilities as u


class TestBinCounts(unittest.TestCase):
    """Test bincounts function"""
    def setUp(self):
        """Make dummy data"""
        self.tso3d = np.ones((4, 256, 2048))
        self.tso4d = np.ones((2, 2, 256, 2048))
        self.bins = lt.wavelength_bins(subarray='SUBSTRIP256')

    def testBin(self):
        # Make a pixel mask
        mask = np.zeros((256, 2048))
        mask[50:100, :] = 1

        # Bin the counts with a mask
        mask_counts = u.bin_counts(self.tso3d, self.bins[0], pixel_mask=mask, plot_bin=10)

        # Bin the counts without a mask
        nomask_counts = u.bin_counts(self.tso3d, self.bins[0])

        # Make sure they are different
        self.assertNotEqual(np.sum(mask_counts), np.sum(nomask_counts))


class TestCombineSpectra(unittest.TestCase):
    """Test combine_spectra function"""
    def setUp(self):
        # Make spectra for testing
        self.s1 = np.array([np.linspace(0.9, 2.8, 2048), np.random.normal(loc=1000, size=2048), np.random.normal(loc=10, size=2048)])
        self.s2 = np.array([np.linspace(0.6, 1.4, 1632), np.random.normal(loc=1000, size=1632), np.random.normal(loc=10, size=1632)])
        self.s3 = np.array([np.linspace(0.5, 0.88, 1632), np.random.normal(loc=1000, size=1632), np.random.normal(loc=10, size=1632)])

    def testOverlap(self):
        """Test that the spectra can be combined when overlapping"""
        # Run function
        result1 = u.combine_spectra(self.s1, self.s2)

        # Check combined spectra result
        self.assertEqual(result1.ndim, 2)
        self.assertEqual(result1.shape[0], 3)

        # # Check if order matters
        # TODO: This fails right now. Fix it!
        # result2 = u.combine_spectra(self.s2, self.s1)
        # self.assertTrue((result1 == result2).all())

    def testNoOverlap(self):
        """Test that the spectra can be combined when overlapping"""
        # Run function
        result1 = u.combine_spectra(self.s1, self.s3)

        # Check combined spectra result
        self.assertEqual(result1.ndim, 2)
        self.assertEqual(result1.shape[0], 3)

        # Check if order matters
        result2 = u.combine_spectra(self.s3, self.s1)
        self.assertTrue((result1 == result2).all())


class TestNanReferencePixels(unittest.TestCase):
    """Test nan_reference_pixels function"""
    def setUp(self):
        """Test instance setup"""
        pass

    def test2D(self):
        """Test the NaNs are inserted into 2D data"""
        # Make 2D data
        data2048_2D = np.ones((2048, 2048))

        # Run the function
        result = u.nan_reference_pixels(data2048_2D)

        # Check shape and NaNs
        self.assertEqual(result.shape, data2048_2D.shape)
        self.assertTrue(np.isnan(result[3, 3]))

    def test3D(self):
        """Test the NaNs are inserted into 3D data"""
        # Make 3D data
        data256_3D = np.ones((4, 256, 2048))

        # Run the function
        result = u.nan_reference_pixels(data256_3D)

        # Check shape and NaNs
        self.assertEqual(result.shape, data256_3D.shape)
        self.assertTrue(np.isnan(result[0, 3, 3]))

    def test4D(self):
        """Test the NaNs are inserted into 4D data"""
        # Make 4D data
        data96_4D = np.ones((2, 2, 96, 2048))

        # Run the function
        result = u.nan_reference_pixels(data96_4D)

        # Check shape and NaNs
        self.assertEqual(result.shape, data96_4D.shape)
        self.assertTrue(np.isnan(result[0, 0, 3, 3]))

    def testRaise(self):
        # Bad data
        bad_shape = np.ones((1, 2, 3, 96, 2048))

        # Throw error if 5+ dimensions
        self.assertRaises(ValueError, u.nan_reference_pixels, bad_shape)


class TestTo3D(unittest.TestCase):
    """Test to_3d function"""
    def setUp(self):
        """Nothing here"""
        pass

    def testIt(self):
        """Test different data shapes"""
        # 2D
        data = np.ones((256, 2048))
        new_data, shape = u.to_3d(data)
        self.assertEqual(len(shape), data.ndim)
        self.assertEqual(new_data.ndim, 3)

        # 3D
        data = np.ones((4, 256, 2048))
        new_data, shape = u.to_3d(data)
        self.assertEqual(len(shape), data.ndim)
        self.assertEqual(new_data.ndim, 3)

        # 4D
        data = np.ones((2, 2, 256, 2048))
        new_data, shape = u.to_3d(data)
        self.assertEqual(len(shape), data.ndim)
        self.assertEqual(new_data.ndim, 3)

        # 1D
        data = np.ones(2048)
        self.assertRaises(ValueError, u.to_3d, data)
