#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 16:24:39 2020

@author: kinger
"""
from fileIO import readRotation, readTranslation
import numpy as np

def getPMatrix(frame):
    file_R = 'data/camera/rotation' + str(frame) + '.txt'
    file_T = 'data/camera/translation' + str(frame) + '.txt'
    # For dataset 2010_03_17_drive_0046
    K = np.array([6.254584e+02, 0.000000e+00, 6.24535e+02, 
               0.000000e+00, 6.254584e+02, 1.911291e+02, 
               0.000000e+00, 0.000000e+00, 1.000000e+00]).reshape(3,3) 
    R = readRotation(file_R)
    T = readTranslation(file_T)
    R = np.vstack((R, np.zeros((1,3))))
    T = np.vstack((T,np.ones((1,1))))
    RT = np.hstack((R, T))
    H = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
    P = K @ H @ RT
    return P