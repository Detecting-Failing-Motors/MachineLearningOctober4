#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 21:43:04 2019

@author: tbryan
"""

import MainHeader as mh



#Save Array Data to user specified director
filename = "TRAININGDATA.csv"
directory = mh.os.listdir(mh.DirectoryName)
mh.os.chdir(mh.DirectoryName)
i = 0
TD = []
for file in directory:
    if ".csv" in file and file != filename:
        dataset = mh.pd.read_csv(file,header = 0)
        t = mh.np.transpose(dataset.values[:,0])
        t = t.astype(float)
        comb_sig = mh.np.transpose(dataset.values[:,1])
        comb_sig = comb_sig.astype(float)
                 
        stats = mh.GetSingleValueStatisticalInfo(comb_sig,mh)
        
        fft = mh.FourierTransform(t,comb_sig,mh.Tmax,mh)
        max_index, max_value = max(enumerate(fft["Freq. Amp."]), key=mh.operator.itemgetter(1))
        
        freqmax = fft["Frequency"][max_index]
        
        if freqmax > 150:
            State = "Good"
        else:
            State = "Bad" 
        
        RawDataML = {
            "State": State,
            "BBFO": 7,
            "Freq. @ Max Ampl.": freqmax
            }
        RawDataML = {**RawDataML, **stats}
        RawDataML = mh.pd.DataFrame(RawDataML, index=[0])
        Title = RawDataML.columns
        RawDataML = RawDataML.values[0,:]
        TD.append(RawDataML)
        
        i += 1
      

Data1 = mh.pd.DataFrame(TD,columns = Title)
open(filename,"w")
Data1.to_csv(filename,index = False)

mh.os.chdir(mh.HomeDirectory)