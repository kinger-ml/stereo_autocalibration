# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 11:33:24 2020

@author: Krishna Kinger
"""

import cv2
import time
from stereo_utilities import stereoMatching, getCorrespondingPoints, getAkazeCorrespondingPoints, computeCommonMatches, removeOutliers
from stereo_utilities import updateIndexList, threeDProject, createPointMap, updatePointMap
from fileIO import saveResults, writePointMap, writeToFile, writeList
from create3DMap import createMap
from results import generateResults
import subprocess
import numpy as np
from prettytable import PrettyTable
import statistics as st
from plotMatches import saveMatches

#Call Orb and Akaze cosntructors
orb = cv2.ORB_create()
akaze = cv2.AKAZE_create()
orb_factor = 0.2    #Factor to get good matches. 0.3 means top thirty percent matches will be selected
#track_frames = 2    #Track features over these number of frames.
total_frames = 100  #Total number of frames for which the algorithm is to be run
start = 0             
method = 'akaze'    #The method for keypoint and descriptor detection.
time_extract = []
time_features = []
time_sba = []
index_list, src, dst, pointMap, visMap = [], [], [], [], [] 
s = "{:06d}".format(start)
iml_base = 'data/2010_03_17_drive_0046/I1_' + s + '.png'
imr_base = 'data/2010_03_17_drive_0046/I2_' + s + '.png'
lImage_base = cv2.cvtColor(cv2.imread(iml_base), cv2.COLOR_BGR2GRAY)
rImage_base = cv2.cvtColor(cv2.imread(imr_base), cv2.COLOR_BGR2GRAY)
kpl_base, desl_base, kpr_base, desr_base, src_base, dst_base = stereoMatching(lImage_base, rImage_base, method)
#index_list.append([1]*len(src_base))
for fr in range(start+1, total_frames):
    start_time = time.time()
    s = "{:06d}".format(fr)
    iml_cur = 'data/2010_03_17_drive_0046/I1_' + s + '.png'
    imr_cur = 'data/2010_03_17_drive_0046/I2_' + s + '.png'
    lImage_cur = cv2.cvtColor(cv2.imread(iml_cur), cv2.COLOR_BGR2GRAY)
    rImage_cur = cv2.cvtColor(cv2.imread(imr_cur), cv2.COLOR_BGR2GRAY)
    time_extract.append(time.time()-start_time)
    start_time = time.time()
    kpl_cur, desl_cur, kpr_cur, desr_cur, src_cur, dst_cur = stereoMatching(lImage_cur, rImage_cur, method)
    #Get matches from base to current frame
    if method == 'orb':
        left_base, left_cur = getCorrespondingPoints(kpl_base, kpl_cur, desl_base, desl_cur, orb_factor)
    if method == 'akaze':
        left_base, left_cur = getAkazeCorrespondingPoints(kpl_base, kpl_cur, desl_base, desl_cur)
       
    #Compute indices of common_points in base and cur frame
    list_base, list_cur = computeCommonMatches(src_base, src_cur, left_base, left_cur)
    #list_base, list_cur = removeOutliers(list_base, list_cur, src_base, src_cur, 50) 
    time_features.append(time.time()-start_time)
    index_list.append(list_cur)

    threeD = threeDProject(src_base, dst_base, 2*(fr-1)).tolist()
    #Write src_base, src_cur, dst_base, dst_cur, visibility
    writeToFile(threeD, src_base, dst_base, src_cur, dst_cur, list_base, list_cur, fr-1) #last value is the frame number which is projected

    kpl_base, desl_base, kpr_base, desr_base, src_base, dst_base = kpl_cur, desl_cur, kpr_cur, desr_cur, src_cur, dst_cur
    lImage_base, rImage_base = lImage_cur, rImage_cur
    
    print('Calling Sparse Bundle Adjustment for frame', fr, 'with ', len(threeD), 'points')
    start_time = time.time()
    popen = subprocess.Popen('./sba2d', stdout=subprocess.PIPE)
    popen.wait()
    time_sba.append(time.time()-start_time)
    output = popen.stdout.read()
    print(output)
    
np.save('data/results/index_list.npy', index_list)
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