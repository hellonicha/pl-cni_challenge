'''
LSTM network for ROI timeseries with demographic data

Author: Nicha C. Dvornek
Date: March 2018
'''

from __future__ import print_function
import numpy as np
import tensorflow as tf
import random as rn
import os

np.random.seed(825)  # for reproducibility, was 82515
rn.seed(825)
tf.set_random_seed(825)

from keras.models import load_model
import scipy.io
import configparser
from numpy import genfromtxt
import glob

def get_subsequences(X,subind,timesteps,demo):
    totalseq = sum(map(len,X[:,0])) - len(X)*(timesteps-1)

    print('Total number of subsequences: ', totalseq)
    Xsubseq = np.empty((totalseq,1),dtype=np.object)
    subindsubseq = np.zeros(totalseq)
    demosubseq = np.zeros((totalseq,demo.shape[1]))
    ind = 0
    for i in range(X.shape[0]):
        temp = X[i,0]
        for j in range(temp.shape[0]-timesteps+1):
            Xsubseq[ind,0] = temp[j:j+timesteps,:]
            subindsubseq[ind] = subind[i]
            demosubseq[ind,:] = demo[i,:]
            ind = ind + 1
    return Xsubseq, subindsubseq, demosubseq

def standardize(timeseries):
    m = np.mean(timeseries,axis=0)
    s = np.std(timeseries,axis=0)
    return (timeseries - m) / s

def read_CNI_data(datadir,datafile):

    subdirs = next(os.walk(datadir))[1]
    X = np.empty((len(subdirs), 1), dtype=np.object)
    demo = np.zeros((len(subdirs),4))
    for i in range(len(subdirs)):
        ts = genfromtxt(os.path.join(datadir,subdirs[i],datafile),delimiter=',')
        X[i,0] = standardize(ts.transpose())

        d = genfromtxt(os.path.join(datadir,subdirs[i],'phenotypic.csv'),dtype=None,delimiter=',',names=True)
        if 'M' in str(d['Sex']):
            demo[i,0] = 1
        demo[i,1] = d['Age']
        demo[i,2] = d['Edinburgh_Handedness']
        demo[i,3] = d['WISC_FSIQ']

    return X, demo

def predict_CNI(inputdir,outputname):

    dirname = os.path.dirname(os.path.abspath(__file__))

    print('Loading data...')
    configfile = dirname + '/config_predict.ini'
    config = configparser.ConfigParser()
    config.read(configfile)
    traindata = config['io']['traindata']
    datafile = config['io']['datafile']
    timesteps = config.getint('model','timesteps')

    X, demo = read_CNI_data(inputdir, datafile)
    subind = np.arange(len(X))
    print('X shape: ', X.shape)

    selectinds = range(X[0, 0].shape[1])
    trainmat = scipy.io.loadmat(dirname + '/' + traindata)
    if 'bestinds' in trainmat:
         selectinds = np.squeeze(np.array(trainmat['bestinds']))
    elif 'inds' in trainmat:
        selectinds = np.squeeze(np.array(trainmat['inds']))
    print(len(selectinds))

    # load and preprocess demographic vars, scaling [min,max] to [-1,1] and repeating values for augmented data
    maxvals = np.squeeze(trainmat['demomax'])
    minvals = np.squeeze(trainmat['demomin'])

    for i in range(demo.shape[1]):
        maxval = maxvals[i]
        minval = minvals[i]
        demo[:, i] = (demo[:, i] - minval) / (maxval - minval) * 2 - 1
        print(demo[:,i])

    X, subind, demo_test = get_subsequences(X, subind, timesteps, demo)
    X_test = np.zeros((len(X),X[0,0].shape[0],len(selectinds)))
    for i in range(len(X)):
        temp = X[i, 0]
        X_test[i, :, :] = temp[None, :, selectinds]

    models = glob.glob((dirname + '/*.hdf5'))
    sinds = np.unique(subind)
    p_subs = np.zeros((len(models),len(sinds)))
    p_seqs = np.zeros((len(models),len(X_test)))
    print(models)
    for m in range(len(models)):
        model = load_model(models[m])
        p = model.predict([X_test, demo_test])

        p_sub = np.zeros(len(sinds))
        for s in range(len(sinds)):
            p_sub[s] = np.mean(p[subind==sinds[s]] > 0.5)
        p_seqs[m, :] = np.squeeze(p)
        p_subs[m, :] = p_sub


    # Ensemble result
    p_sub_ens = np.mean(p_subs,axis=0)

    final_class = p_sub_ens > 0.5
    print(sum(final_class))
    np.savetxt(outputname,final_class,fmt='%d')