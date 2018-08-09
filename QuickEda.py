#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 15:59:42 2018

@author: raz
"""

import numpy as np
import pandas as pd


class QuickEda:

    """
    Class Doc is here
    """

    def __init__(self, data):
        self.data = data

    def missing_analysis(self, opts=None):
        """ find missing data for each feature"""
        pass

    def feature_types(self, cat_level):
        """ seperate categorical and continuous features, optionally use a
        threshold value to separate categorical from continuous
        """
        pass

    def check_correlations(self, opts=None):
        """ find highly correlated feature pairs in data, threshhold could be optional """
        pass

    def plot(self, kind=None):
        "yellow"
