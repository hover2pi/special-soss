#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `summation` module."""

import unittest
from pkg_resources import resource_filename

import numpy as np

from specialsoss import summation as sm


class TestSummation(unittest.TestCase):
    """Test functions in summation.py"""
    def setUp(self):
        """Test instance setup"""
        pass

    def test_extract(self):
        """Test for extract function"""
        # Filters and subarays
        filters = 'CLEAR', 'F277W'
        ypix = 96, 256, 2048
        subarrays = 'SUBSTRIP256', 'SUBSTRIP96', 'FULL'

        for filt in filters:
            for subarray, pix in zip(subarrays, ypix):

                # Data
                data = np.ones((2, 2, pix, 2048))

                # Run the extraction
                result = sm.extract(data, filt=filt, subarray=subarray)