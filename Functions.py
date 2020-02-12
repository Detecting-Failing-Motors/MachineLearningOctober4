#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 14:33:39 2019

@author: tbryan
"""

def FourierTransform(t,comb_sig,Tmax,mh):
    #Fast Fourier Transform
    number_of_time_samples = len(t)
    frq = mh.np.arange(number_of_time_samples)/(Tmax)# two sides frequency range
    frq = frq[range(int(number_of_time_samples/(2)))] # one side frequency range
    Y = abs(mh.np.fft.fft(comb_sig))/number_of_time_samples # fft computing and normalization
    Y = Y[range(int(number_of_time_samples/2))]
    k = mh.np.arange(len(Y))
    #End fft
    x = {
        "Frequency":frq,
        "Freq. Amp.": Y
        }
    return x

def GetSingleValueStatisticalInfo(comb_sig,mh):
    #Gather statistical info on the data
    kurtosis_value = mh.kurtosis(comb_sig)
    skew_value = mh.skew(comb_sig)
    mean_value = mh.np.mean(comb_sig)
    max_value = mh.np.max(comb_sig)
    min_value = mh.np.min(comb_sig)
    
    x = {
        "Kurtosis":kurtosis_value,
        "Skew":skew_value,
        "Mean": mean_value,
        "Max":  max_value,
        "Min": min_value
        }
    #Additional way
    #x.update({"Skew":skew_value})
    return x


def SignalGenerator(t,mh):
    #Signal generator for practice
    noise1 = mh.np.random.randn(len(t))                # white noise 1
    noise2 = mh.np.random.randn(len(t))                # white noise 2 
    noise3 = mh.np.random.randn(len(t))                # white noise 3
    phase1 = 0;                                     #radians
    phase2 = mh.np.pi / 2                              #radians
    phase3 = -1 * mh.np.pi /7                         #radians
    frequency1 =  11                              #Hz
    frequency2 = 9                              #Hz
    frequency3 = 67                               #Hz
    mag1 = 2
    mag2 = 3
    mag3 = 4
    base1 = mag1 * mh.np.sin(2 * mh.np.pi * frequency1 * t + phase1 ) + noise1  #base signal
    base2 = mag2 * mh.np.sin(2 * mh.np.pi * frequency2 * t + phase2 ) + noise2  #base signal
    base3 = mag3 * mh.np.sin(2 * mh.np.pi * frequency3 * t + phase3 ) + noise3  #base signal
    return base1 + base2 + base3
    #End of practice signal generator
    
def PlotFrequency(x,y,i,mh):
    mh.plt.close('all')
    fig = mh.plt.figure(i)
    mh.plt.plot(x,y,'r')
    mh.plt.xlabel('Freq (Hz)')
    mh.plt.ylabel('Magnitude')
    mh.plt.title('Spectrum')
    mh.plt.grid(True)
    return fig
  
def PlotAmplitude(x,y,i,mh):
    mh.plt.close('all')
    fig = mh.plt.figure(i)
    mh.plt.plot(x,y,'b')
    mh.plt.xlabel('Time (s)')
    mh.plt.ylabel('Amplitude')
    mh.plt.title('Raw Data')
    mh.plt.grid(True)
    return fig
    