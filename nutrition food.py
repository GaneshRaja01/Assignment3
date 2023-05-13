# -*- coding: utf-8 -*-
"""
Created on Fri May 12 20:23:18 2023

@author: rgane
"""

# Importing the required libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cluster_tools as ct
import sklearn.cluster as cluster
import sklearn.metrics as skmet
import scipy.optimize as opt


def clean_df(df, time, class_name):
    """
    Cleans a pandas dataframe by filtering rows based on the specified time 
    and classification name.
    """
    df1 = df[df['Time'] == time]
    df2 = df1[df1['Classification Name'] == class_name]
    return df2


def remove_cols(df, cols):
    """
    Remove specified columns from a pandas DataFrame.
    """
    df.drop(cols, axis=1, inplace=True)
    return df


def line(x, m, c):
    """
    Compute the values of a linear function at the specified input points.
    """
    y = (m * x) + c
    return y


