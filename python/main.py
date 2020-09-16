# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 17:16:07 2020

@author: Krishna Kinger

Code Flow: Take Frame 0 and 1. Run SBA. Take Frame 2. Project point from 0 and 1 to 3D. Prepare SBA files and call SBA.
"""

import cv2
import time
from stereo_utilities import stereoMatching, getCorrespondingPoints, getAkazeCorrespondingPoints, computeCommonMatches, removeOutliers
from stereo_utilities import updateIndexList, threeDProject, createPointMap, updatePointMap
from fileIO import saveResults, writePointMap
import subprocess
from create3DMap import createMap
import numpy as np
from prettytable import PrettyTable
import statistics as st
from results import generateResults

#Call Orb and Akaze cosntructors
orb = cv2.ORB_create()
akaze = cv2.AKAZE_create()
orb_factor = 0.4    #Factor to get good matches. 0.3 means top thirty percent matches will be selected
track_frames = 3    #Track features over these number of frames.
total_frames = 100   #Total number of frames for which the algorithm is to be run
ite = 0             
method = 'akaze'    #The method for keypoint and descriptor detection. 
vis_list = []
time_extract = []
time_features = []
time_sba = []
while ite < total_frames:
    index_list, src, dst, pointMap, visMap = [], [], [], [], []
    s = "{:06d}".format(ite)
    iml_base = 'data/2010_03_17_drive_0046/I1_' + s + '.png'
    imr_base = 'data/2010_03_17_drive_0046/I2_' + s + '.png'
    start_time = time.time()
    lImage_base = cv2.cvtColor(cv2.imread(iml_base), cv2.COLOR_BGR2GRAY)
    rImage_base = cv2.cvtColor(cv2.imread(imr_base), cv2.COLOR_BGR2GRAY)
    time_extract.append(time.time()-start_time)
    start_time = time.time()
    kpl_base, desl_base, kpr_base, desr_base, src_base, dst_base = stereoMatching(lImage_base, rImage_base, method)
    time_features.append(time.time()-start_time)
    src.append(src_base)
    dst.append(dst_base)
    for fr in range(ite+1, ite+track_frames):
        s = "{:06d}".format(fr)
        iml_cur = 'data/2010_03_17_drive_0046/I1_' + s + '.png'
        imr_cur = 'data/2010_03_17_drive_0046/I2_' + s + '.png'
        start_time = time.time()
        lImage_cur = cv2.cvtColor(cv2.imread(iml_cur), cv2.COLOR_BGR2GRAY)
        rImage_cur = cv2.cvtColor(cv2.imread(imr_cur), cv2.COLOR_BGR2GRAY)
        time_extract.append(time.time()-start_time)
        start_time = time.time()
        kpl_cur, desl_cur, kpr_cur, desr_cur, src_cur, dst_cur = stereoMatching(lImage_cur, rImage_cur, method)
        src.append(src_cur)
        dst.append(dst_cur)
        #Get matches from base to current frame
        if method == 'orb':
            left_base, left_cur = getCorrespondingPoints(kpl_base, kpl_cur, desl_base, desl_cur, orb_factor)
        if method == 'akaze':
            left_base, left_cur = getAkazeCorrespondingPoints(kpl_base, kpl_cur, desl_base, desl_cur)
        #Compute indices of common_points in base and cur frame
        list_base, list_cur = computeCommonMatches(src_base, src_cur, left_base, left_cur)
        #list_base, list_cur = removeOutliers(list_base, list_cur, src_base, src_cur, 100)
        kpl_base, desl_base, kpr_base, desr_base, src_base, dst_base = kpl_cur, desl_cur, kpr_cur, desr_cur, src_cur, dst_cur
        if fr == ite+1:
            index_list.append(list_base)
            index_list.append(list_cur)
            pointMap, visMap = createPointMap(src, dst, list_base, list_cur)
            vis_list.append(list_cur)
        else:
            index_list = updateIndexList(index_list, list_base, list_cur)  
            pointMap, visMap = updatePointMap(src, dst, list_base, list_cur, pointMap, visMap, index_list)    
        time_features.append(time.time()-start_time)

        writePointMap(pointMap, visMap, ite, fr)
        start_time = time.time()
        print('Calling Sparse Bundle Adjustment with ',len(pointMap),' cameras and ', len(pointMap[0]), 'points')
        popen = subprocess.Popen('./sba2d', stdout=subprocess.PIPE)
        popen.wait()
        time_sba.append(time.time()-start_time)
        output = popen.stdout.read()
        print(output)
    #saveResults method save the matching of consecutive frames
    #saveResults(src,dst, index_list, ite)  
    ite = ite+1
#index_list stores the visibility from each bundle adjustmet
#create map uses this list to plot the 3D point cloud and save it as ply file
np.save('data/results/index_list.npy', vis_list)
createMap()

results = generateResults(total_frames -1)
print('Performed stereo autocalibration for ', str(total_frames), ' frames.')
t = PrettyTable(['Extrinsic Component', 'Mean Error'])
t.add_row(['Rotation_x', results[0]])
t.add_row(['Rotation_y', results[1]])
t.add_row(['Rotation_z', results[2]])
t.add_row(['Translation_x', results[3]])
t.add_row(['Translation_y', results[4]])
t.add_row(['Translation_z', results[5]])
print(t)
t = PrettyTable(['Process', 'Mean Time'])
t.add_row(['Image Retrieval', st.mean(time_extract)])
t.add_row(['Feature Extraction/Matching', st.mean(time_features)])
t.add_row(['Sparse Bundle Adjustment', st.mean(time_sba)])
print(t)