#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 15:59:42 2018

@author: raz
"""
import re
import datetime

import numpy as np
import pandas as pd


class QuickEda:

    """
    Class Doc is here
    """

    def __init__(self, data):
        assert isinstance(data, pd.DataFrame)
        self.data = data
        self.catfeatures = []
        self.numfeatures = []

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
        pass

    def best_features(self, xfeat, yfeat):
        """return top n features, try regression/logistic model"""
        pass

    def isdate(self, datestring):
        """
        s
        """
        try:
            mat = re.match('(\d{2})[/.-](\d{2})[/.-](\d{4})$', datestring)
            if mat is not None:
                datetime.datetime(*(map(int, mat.groups()[-1::-1])))
                return True
        except ValueError:
            pass
        return False

    def get_dtype(self, xstr):
        """# get data type with regex (is float?)
        """
        if re.match("^\d+?\.\d+?$", xstr) is None:
            if xstr.isdigit():
                return "int"
            else:
                if self.isdate(xstr):
                    return "date"
                return "string"
        return "float"

    def process_dtypes(self, data):
        """read data, and assign data types accordingly
        """

        for col, x in data.iloc[1, :].iteritems():
            t = self.get_dtype(str(x).strip())
            if(t == "int" or t == "float"):
                self.numfeatures.append(col)
                if t == "int":
                    data[col] = data[col].astype(int)
                else:
                    data[col] = data[col].astype(float)
            else:
                self.catfeatures.append(col)
                data[col] = data[col].astype(str)
        return data
    
    def plot_learning_curve(data, clf):
        pass