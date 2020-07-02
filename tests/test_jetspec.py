#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `jetspec` module."""

import unittest
from pkg_resources import resource_filename

import numpy as np

from specialsoss import jetspec


class TestJetspec(unittest.TestCase):
    """Test functions in jetspec.py"""
    def setUp(self):
        """Test instance setup"""
        # Get files for testing
        self.frame = np.ones((256, 2048))
        self.tso3d = np.ones((4, 256, 2048))
        self.tso4d = np.ones((2, 2, 256, 2048))
