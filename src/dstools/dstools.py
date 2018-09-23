#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 15:59:42 2018

@author: raz
"""
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, RobustScaler


class DStools:

    """
    Class Doc is here
    """

    def __init__(self):
        self.features = None
        pass

    def missing_analysis(self, opts=None):
        """ find missing data for each feature"""
        pass

    def feature_types(self, cat_level):
        """ seperate categorical and continuous features, optionally use a
        threshold value to separate categorical from continuous
        """
        pass

    @staticmethod
    def check_correlations(data, numfeatures, t=.90, plot=True):
        """ find highly correlated feature pairs in data, threshhold could be optional """
        data_cor = data[numfeatures].corr()
        # Set the threshold and add to pairs to list

        if plot:
            # %matplotlib inline
            sns.set(rc={'figure.figsize': (
                len(numfeatures)*1.5, len(numfeatures))})

            mask = np.zeros_like(data_cor, dtype=np.bool)
            mask[np.triu_indices_from(mask)] = True

            # Set up the matplotlib figure
            # Generate a custom diverging colormap
            cmap = sns.diverging_palette(220, 10, as_cmap=True)

            # Draw the heatmap with the mask and correct aspect ratio
            sns.heatmap(data_cor, mask=mask, cmap=cmap, vmax=.7,
                        center=0, linewidths=.8, cbar_kws={"shrink": .5})

        cor_list = []
        # Find Pairs and update cor_list
        for i in range(len(numfeatures)-1):
            for j in range(i+1, len(numfeatures)):
                if (abs(data_cor.iloc[i, j]) >= t):
                    cor_list.append([data_cor.iloc[i, j], j, i])

        if cor_list:
            # Sort by corr coef
            sorted_list = sorted(cor_list, key=lambda x: -abs(x[0]))
            for v, i, j in sorted_list:
                print("%s and %s = %.5f" %
                      (numfeatures[:][i], numfeatures[:][j], v))
        else:
            print("None of the features have correlation higher than ", t)

    @staticmethod
    def dist_plots(data, numfeatures, scale=True):
        "Generate dist plots for provided feature vector"
        sns.set(font_scale=1)
        temp_df = data[numfeatures].fillna(data[numfeatures].median())

        if scale:
            scaler = MinMaxScaler()
            scaler.fit(temp_df)
            temp_df = pd.DataFrame(scaler.transform(
                temp_df), columns=numfeatures)

        temp_df = temp_df.melt(var_name='feature',  value_name='values')

        graph = sns.FacetGrid(temp_df, col='feature', height=4, aspect=1)
        graph = (graph.map(sns.distplot, "values", kde=True, rug=False, fit=stats.gamma,
                           color="green", kde_kws={"color": "r", "lw": 3, "label": "fitted"}))
        del temp_df

    @staticmethod
    def count_plots(data, catfeatures, xhue=None):
        "Generate conut plots for provided feature vector"
        tmp_df = data[catfeatures].copy()
        plt.figure(figsize=(len(catfeatures)*7, len(catfeatures)*7*.75))
        sns.set(font_scale=1.5)
        if xhue:
            catfeatures.remove(xhue)
        for i, col in enumerate(catfeatures):

            plt.subplot(len(catfeatures), 3, i+1)
            if xhue:
                sns.countplot(x=col, data=tmp_df, palette="Set2", hue=xhue,
                              order=data[col].value_counts().index).set(xlabel=col)
            else:
                sns.countplot(x=col, data=tmp_df, palette="Set2",
                              order=data[col].value_counts().index).set(xlabel=col)

        plt.tight_layout()
        plt.show()
        plt.clf()
        plt.close('all')

    def best_features(self, xfeat, yfeat):
        """return top n features, try regression/logistic model"""
        pass

    def _isdate(self, datestring):
        """
        s
        """
        mat = re.match('(\d{1,2})[/.-](\d{1,2})[/.-](\d{2,4})$', datestring)

        try:
            if mat:
                # xdate = datetime.datetime(*(map(int, mat.groups()[-1::-1])))
                return True
            else:
                return False
        except ValueError:
            print("err - isdate")

    def _get_dtype(self, xstr):
        """# get data type with regex (is float?)
        """
        if re.match("^\d+?\.\d+?$", xstr):
            return "float"
        elif xstr.isdigit():
            return "int"
        elif self._isdate(xstr):
            return "date"
        else:
            return "string"

    def process_dtypes(self, data, tapply=False, thr=10):
        """read data, and assign data types accordingly
        """

        features = defaultdict(list)
        n = len(data)

        for col, val in data.iloc[3, :].iteritems():
            val = str(val).strip()
            val_type = self._get_dtype(val)
            unq = data[col].nunique()

            if unq > n*.95:
                features["skip"].append(col)
                continue
            try:
                if(val_type == "int" or val_type == "float"):
                    if unq >= thr:
                        features["numfeatures"].append(col)
                        if tapply:
                            if val_type == "int" and data[col].dtype != 'int':
                                data[col] = data[col].astype(np.int64)
                            elif val_type == 'float' and data[col].dtype != 'float':
                                data[col] = data[col].astype(np.float64)
                    else:
                        features["catfeatures"].append(col)
                        if tapply:
                            print("feature ", col, " contains ", unq,
                                  " unique values, converted to categorical encoding")
                            data[col] = data[col].astype('category')
                elif val_type == "string":
                    if unq <= thr:
                        features["catfeatures"].append(col)
                        if tapply:
                            data[col] = data[col].astype('category')
                    else:
                        if tapply:
                            print("feature ", col, " contains ", unq,
                                  " unique values, converted to numeric encoding")
                            data[col] = data[col].astype('category')
                            data[col] = data[col].cat.codes
                            features["encode"].append(col)
                        else:
                            features["encode"].append(col)

                elif val_type == "date":
                    features["dtfeatures"].append(col)
                    if tapply:
                        data[col] = pd.to_datetime(data[col])

            except ValueError as verror:
                print(verror, "\n Data Contains N/A values in :", col)

        self.features = features

        return features

    def plot_learning_curve(self, data, clf):
        pass
