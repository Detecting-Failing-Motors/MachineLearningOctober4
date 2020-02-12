#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 11:05:34 2019

@author: tbryan
"""

#Import Common Place Libraries
import matplotlib.pyplot as plt
import numpy as np 
from scipy.stats import kurtosis
from scipy.stats import skew
from scipy.signal import hilbert
import os
import pandas as pd
from datetime import datetime
import operator

#Import Created Functions
from Functions import FourierTransform
from Functions import GetSingleValueStatisticalInfo
from Functions import SignalGenerator
from Functions import PlotFrequency
from Functions import PlotAmplitude

#Import Machine Learning Libraries
from pandas.plotting import scatter_matrix
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

#Define global variables
global SampleFrequency, FrequencyResolution, dt,Tmax, t
global HomeDirectory, DirectoryName
SampleFrequency = 2000
FrequencyResolution = 0.1
dt = 1/SampleFrequency
Tmax = 1 / FrequencyResolution
t = np.arange(0,Tmax,dt)
HomeDirectory = os.getcwd()
DirectoryName = 'Data'