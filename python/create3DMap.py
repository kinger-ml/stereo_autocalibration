#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 21:22:18 2020

@author: kinger
"""
import numpy as np
import sys
from fileIO import read3dpoints


ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
end_header
'''
#property uchar red
#property uchar green
#property uchar blue

def writeply(txtfile):
    nums = []
    with open(txtfile, 'r') as f:
        for line in f:
            s = line.split()
            #nums.append(tuple(map(lambda x: float(x), s)))
            nums.append(list(map(lambda x: float(x), s)))
    verts = np.asarray(nums)
    verts = verts.reshape(-1, 3)
    #colors = colors.reshape(-1, 3)
    #verts = np.hstack([verts, colors])
    plyfile = txtfile[:-3] + "ply"
    with open(plyfile, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        #np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')
        np.savetxt(f, verts, fmt='%f %f %f')

def createMap():
    index_list = np.load('data/results/index_list.npy', allow_pickle=True)
    threeD, indices = [], []
    for i in range(0, len(index_list)):
        indices.append(index_list[i])
    
    threeD_init = read3dpoints('data/results/threeD0.txt')
    threeD_init = threeD_init.tolist() 
    threeD.append(threeD_init)
    for i in range(1,len(indices)-1):
        file = 'data/results/threeD'+str(i)+'.txt'
        threeD_temp = read3dpoints(file)
        threeD_temp = threeD_temp.tolist()
        for index in indices[i-1]:
            threeD[0].append(threeD_temp[index])
            
    threeD_np = np.array(threeD)
    threeD_np = threeD_np.reshape(len(threeD_np[0]), 3)
    np.savetxt('data/results/threeD.txt', threeD_np)
    writeply('data/results/threeD.txt')

#createMap()
    
    
    