#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `summation` module."""

import unittest
from pkg_resources import resource_filename

from specialsoss import summation as sm


class TestSummation(unittest.TestCase):
    """Test functions in summation.py"""
    def setUp(self):
        """Test instance setup"""
        # Make dummy data
        self.tso4d = np.ones((2, 2, 256, 2048))

    def test_extract(self):
        """Test for extract function"""
        # Filters and subarays
        filters = 'CLEAR', 'F277W'
        subarrays = 'SUBSTRIP256', 'SUBSTRIP96', 'FULL'

        for filt in filters:
            for subarray in subarrays:

                # Run the extraction
                result = sm.extract(self.tso4d, filt=filt, subarray=subarray)