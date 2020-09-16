#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 20:12:00 2020

@author: kinger
"""
import numpy as np
import matplotlib.pyplot as plt
import statistics as st
def readRotation(filename):
    file = open(filename, 'r')
    str = ''
    for i in range(1,7):
        file.readline()
    str+=file.read()
    str = str.replace('data: [ ','')
    str = str.replace(']','')
    str = str.replace(',',' ')
    rotation = str.split()
    rotation = np.array(rotation, dtype=np.float32)
    file.close()
    rotation = rotation.reshape(1,3)
    return rotation

def readTranslation(filename):
    file = open(filename, 'r')
    str = ''
    for i in range(1,7):
        file.readline()
    str+=file.read()
    str = str.replace('data: [ ','')
    str = str.replace(']','')
    str = str.replace(',',' ')
    translation = str.split()
    translation = np.array(translation, dtype=np.float32).reshape(3,1)
    file.close()
    return translation

def generateResults(frames):
    Rx, Ry, Rz = [], [], []
    Rx_l, Ry_l, Rz_l = [], [], []
    Rx_r, Ry_r, Rz_r = [], [], []
    for cam in range(0, frames):
        file_left = 'data/camera/rotation' + str(2*cam) + '.txt'
        file_right = 'data/camera/rotation' + str((2*cam)+1) + '.txt'
        R_left = readRotation(file_left)
        R_right = readRotation(file_right)
        Rx.append(R_right[0][0] - R_left[0][0])
        Ry.append(R_right[0][1] - R_left[0][1])
        Rz.append(R_right[0][2] - R_left[0][2])  
        Rx_l.append(R_left[0][0])
        Ry_l.append(R_left[0][1])
        Rz_l.append(R_left[0][2])
        Rx_r.append(R_right[0][0])
        Ry_r.append(R_right[0][1])
        Rz_r.append(R_right[0][2])
    plt.plot(Rx, label = 'Rx')
    plt.plot(Ry, label = 'Ry')
    plt.plot(Rz, label = 'Rz')
    axes = plt.gca()
    axes.set_ylim([-0.1,0.1])
    plt.legend(loc='best')
    plt.savefig('data/plots/relativeRotation.png')
    plt.clf()
    plt.plot(Rx_l, label = 'Left Rx')
    plt.plot(Ry_l, label = 'Left Ry')
    plt.plot(Rz_l, label = 'Left Rz')
    plt.plot(Rx_r, label = 'Right Rx')
    plt.plot(Ry_r, label = 'Right Ry')
    plt.plot(Rz_r, label = 'Right Rz')
    plt.legend(loc='best')
    plt.savefig('data/plots/leftrightRotation.png')
    plt.clf()
    
    Tx, Ty, Tz = [], [], []
    Tx_l, Ty_l, Tz_l = [], [], []
    Tx_r, Ty_r, Tz_r = [], [], []
    for cam in range(0, frames):
        file_left = 'data/camera/translation' + str(2*cam) + '.txt'
        file_right = 'data/camera/translation' + str((2*cam)+1) + '.txt'
        T_left = readTranslation(file_left)
        T_right = readTranslation(file_right)
        Tx.append(T_right[0][0] - T_left[0][0])
        Ty.append(T_right[1][0] - T_left[1][0])
        Tz.append(T_right[2][0] - T_left[2][0])  
    
        Tx_l.append(T_left[0][0])
        Ty_l.append(T_left[1][0])
        Tz_l.append(T_left[2][0])
        Tx_r.append(T_right[0][0])
        Ty_r.append(T_right[1][0])
        Tz_r.append(T_right[2][0])
    
    plt.plot(Tx, label = 'Tx')
    plt.plot(Ty, label = 'Ty')
    plt.plot(Tz, label = 'Tz')
    axes = plt.gca()
    axes.set_ylim([-1,2])
    plt.legend(loc='best')
    plt.savefig('data/plots/relativeTranslation.png')
    plt.clf()
    plt.plot(Tx_l, label = 'Left Tx')
    plt.plot(Ty_l, label = 'Left Ty')
    plt.plot(Tz_l, label = 'Left Tz')
    plt.plot(Tx_r, label = 'Right Tx')
    plt.plot(Ty_r, label = 'Right Ty')
    plt.plot(Tz_r, label = 'Right Tz')
    plt.legend(loc='best')
    plt.savefig('data/plots/leftrightTranslation.png')
    plt.clf()
    results = [np.mean(Rx), np.mean(Ry), st.mean(Rz), st.mean(Tx),st.mean(Ty),st.mean(Tz)]
    return results

if __name__== "__main__":
    results = generateResults(100)