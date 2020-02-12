#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 14:30:47 2019

@author: tbryan
"""
import MainHeader as mh


program_state = 1

while program_state != 0:
    
    #run gui code
    comb_sig = mh.SignalGenerator(mh.t,mh) #eventually will be sensor data    
    stats = mh.GetSingleValueStatisticalInfo(comb_sig,mh)
    fft = mh.FourierTransform(mh.t,comb_sig,mh.Tmax,mh)
    
    #Save Array Data to user specified director
    mh.os.chdir(mh.DirectoryName)
    Data1 = {
        'Time':mh.t,
        'Amplitude':comb_sig,
        'Hilbert': mh.hilbert(comb_sig)
        }
    Data1 = mh.pd.DataFrame(Data1)
    CurrentTime = mh.datetime.now().strftime("%m_%d_%Y__%H_%M_%S")
    open(CurrentTime + '.csv',"w")
    Data1.to_csv(CurrentTime + '.csv',index = False)

    mh.os.chdir(mh.HomeDirectory)
    
    
    #Plot past raw datas
    directory = mh.os.listdir(mh.DirectoryName)
    mh.os.chdir(mh.DirectoryName)
    i = 0
    for file in directory:
        if ".csv" in file and file != "TRAININGDATA.csv":
            dataset = mh.pd.read_csv(file,header = 0)
            t = mh.np.transpose(dataset.values[:,0])
            t = t.astype(float)
            comb_sig = mh.np.transpose(dataset.values[:,1])
            comb_sig = comb_sig.astype(float)
                 
            stats = mh.GetSingleValueStatisticalInfo(comb_sig,mh)
        
            fft = mh.FourierTransform(t,comb_sig,mh.Tmax,mh)
            
            fig = mh.PlotFrequency(fft['Frequency'],fft['Freq. Amp.'],i,mh)
            #mh.plt.show(fig) plots all raw data to console
            i += 1
      
    mh.os.chdir(mh.HomeDirectory)
    #End of raw data plotting
    
    #Plot Current Raw Data in Time Domain
    directory = mh.os.listdir(mh.DirectoryName)
    mh.os.chdir(mh.DirectoryName)
    for file in directory:
        
        if file == (CurrentTime + '.csv'):
            dataset = mh.pd.read_csv(file,header = 0)
            t = mh.np.transpose(dataset.values[:,0])
            t = t.astype(float)
            comb_sig = mh.np.transpose(dataset.values[:,1])
            comb_sig = comb_sig.astype(float)
            fft = mh.FourierTransform(t,comb_sig,mh.Tmax,mh)
            fig1 = mh.PlotFrequency(fft['Frequency'],fft['Freq. Amp.'],i,mh)
            mh.plt.show(fig1)
            fig2 = mh.PlotAmplitude(t,comb_sig,i,mh)
            mh.plt.show(fig2)

    mh.os.chdir(mh.HomeDirectory)
    mh.plt.close('all')
    #End of Plotting Current Raw Data
    
    
    
    #Begin Machine Learning
    
    #Convert Raw Data to Training Data Form
    directory = mh.os.listdir(mh.DirectoryName)
    mh.os.chdir(mh.DirectoryName)
    for file in directory:
        if file == (CurrentTime + '.csv'):
            dataset = mh.pd.read_csv(file,header = 0)
            t = mh.np.transpose(dataset.values[:,0])
            t = t.astype(float)
            comb_sig = mh.np.transpose(dataset.values[:,1])
            comb_sig = comb_sig.astype(float)
            
            stats = mh.GetSingleValueStatisticalInfo(comb_sig,mh)
            fft = mh.FourierTransform(t,comb_sig,mh.Tmax,mh)
    max_index, max_value = max(enumerate(fft["Freq. Amp."]), key=mh.operator.itemgetter(1))

    freqmax = fft["Frequency"][max_index]        
    RawDataML = {
            "State": "Test",
            "BBFO": 7,
            "Freq. @ Max Ampl.": freqmax
            }
    RawDataML = {**RawDataML, **stats}
    RawDataML = mh.pd.DataFrame(RawDataML, index=[0])
    Title = RawDataML.columns
    RawDataML = RawDataML.values[0,1:]

     

    mh.os.chdir(mh.HomeDirectory)
    
    
    #Get Data
    directory = mh.os.listdir(mh.DirectoryName)
    mh.os.chdir(mh.DirectoryName)
    for file in directory:
        if file == 'TRAININGDATA.csv':
            dataset = mh.pd.read_csv(file,header = 0)
    X = dataset.values[:,1:(len(RawDataML)-1)]
    Y = dataset.values[:,0]
    validation_size = 0.20
    seed = 6
    X_train, X_validation, Y_train, Y_validation = mh.model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)        
    
    # Test options and evaluation metric
    scoring = 'accuracy'
    
    # Spot Check Algorithms
    models = []
    #models.append(('LR', mh.LogisticRegression(solver='liblinear', multi_class='ovr')))
    #models.append(('LDA', mh.LinearDiscriminantAnalysis()))
    #models.append(('KNN', mh.KNeighborsClassifier()))
    #models.append(('CART', mh.DecisionTreeClassifier()))
    models.append(('NB', mh.GaussianNB()))
    #models.append(('SVM', mh.SVC(gamma='auto')))
    print(models)
    # evaluate each model in turn
    results = []
    names = []
    for name, model in models:
        	kfold = mh.model_selection.KFold(n_splits=10, random_state=seed)
        	cv_results = mh.model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
        	results.append(cv_results)
        	names.append(name)
        	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        	print(msg)
    
    mh.os.chdir(mh.HomeDirectory)
    
    
    NB = mh.GaussianNB()
    FIT = NB.fit(X_train, Y_train)
    Predictions = NB.predict(X_validation)
    PredictionsProb = NB.predict_proba(X_validation)
    PredictionsProb = PredictionsProb.round(2)
    print(mh.accuracy_score(Y_validation, Predictions))
    print(mh.confusion_matrix(Y_validation, Predictions))
    print(mh.classification_report(Y_validation, Predictions))
    
    #Might use linear regression to get weights to send to microcontroller
    #Could use a weightage of the several machine learning algorithms
    
    
    #End of Machine Learning
    
    
    

    program_state = 0 

    
print('End of File')