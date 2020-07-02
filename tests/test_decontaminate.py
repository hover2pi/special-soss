#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `decontaminate` module."""

import unittest
from pkg_resources import resource_filename

from specialsoss import decontaminate


class TestDecontaminate(unittest.TestCase):
    """Test functions in decontaminate.py"""
    def setUp(self):
        """Test instance setup"""
        # Get files for testing
        self.frame = np.ones((256, 2048))
        self.tso3d = np.ones((4, 256, 2048))
        self.tso4d = np.ones((2, 2, 256, 2048))
