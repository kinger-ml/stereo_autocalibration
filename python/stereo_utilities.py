# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 17:28:22 2020

@author: Krishna Kinger
"""

import cv2
import numpy as np
from math import sqrt
import math  
from camera_params import getPMatrix

def getCorrespondingPoints(kpl, kpr, descriptor1, descriptor2, factor):
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matched_keypoints = matcher.match(descriptor1, descriptor2, None)
    
    # Sort matches by score
    matched_keypoints.sort(key=lambda x: x.distance, reverse=False)
     
    # Remove not so good matches
    numGoodMatches = int(len(matched_keypoints) *factor)
    matched_keypoints = matched_keypoints[:numGoodMatches]
       
    # Extract location of good matches
    points1 = np.zeros((len(matched_keypoints), 2), dtype=np.float32)
    points2 = np.zeros((len(matched_keypoints), 2), dtype=np.float32)
     
    for i, match in enumerate(matched_keypoints):
      points1[i, :] = kpl[match.queryIdx].pt
      points2[i, :] = kpr[match.trainIdx].pt
    
    #source_pts = points1.reshape(1, points1.shape[0], 2)
    #destination_pts = points2.reshape(1, points2.shape[0], 2)
    source_pts = points1.tolist()
    destination_pts = points2.tolist()
    return source_pts, destination_pts

def getCPoints(kpl, kpr, descriptor1, descriptor2, factor):
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matched_keypoints = matcher.match(descriptor1, descriptor2, None)
    
    # Sort matches by score
    matched_keypoints.sort(key=lambda x: x.distance, reverse=False)
     
    # Remove not so good matches
    numGoodMatches = int(len(matched_keypoints) *factor)
    matched_keypoints = matched_keypoints[:numGoodMatches]
       
    # Extract location of good matches
    points1 = np.zeros((len(matched_keypoints), 2), dtype=np.float32)
    points2 = np.zeros((len(matched_keypoints), 2), dtype=np.float32)
     
    for i, match in enumerate(matched_keypoints):
      points1[i, :] = kpl[match.queryIdx].pt
      points2[i, :] = kpr[match.trainIdx].pt
    
    source_pts = points1.reshape(1, points1.shape[0], 2)
    destination_pts = points2.reshape(1, points2.shape[0], 2)
    #source_pts = points1.tolist()
    #destination_pts = points2.tolist()
    return source_pts, destination_pts

def getAkazeCorrespondingPoints(kpl, kpr,  desl, desr):
    p1, p2 = getCPoints(kpl, kpr,  desl, desr, 0.2)
    #p1 = p1.reshape(-1, p1.shape[0], 2)
    #p2 = p2.reshape(-1, p2.shape[0], 2)
    H, _ = cv2.findHomography(p1, p2, cv2.RANSAC, 0.7)
    homography = np.array(H).reshape(3,3)
    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)
    nn_matches = matcher.knnMatch(desl, desr, 2)
    
    matched1 = []
    matched2 = []
    nn_match_ratio = 0.8 # Nearest neighbor matching ratio
    for m, n in nn_matches:
        if m.distance < nn_match_ratio * n.distance:
            matched1.append(kpl[m.queryIdx])
            matched2.append(kpr[m.trainIdx])
    
    inliers1 = []
    inliers2 = []
    good_matches = []
    inlier_threshold = 2.5 # Distance threshold to identify inliers with homography check
    for i, m in enumerate(matched1):
        col = np.ones((3,1), dtype=np.float64)
        col[0:2,0] = m.pt
        col = np.dot(homography, col)
        col /= col[2,0]
        dist = sqrt(pow(col[0,0] - matched2[i].pt[0], 2) +\
                    pow(col[1,0] - matched2[i].pt[1], 2))
        if dist < inlier_threshold:
            good_matches.append(cv2.DMatch(len(inliers1), len(inliers2), 0))
            inliers1.append(matched1[i])
            inliers2.append(matched2[i])
    pts1 = [list(inliers1[idx].pt) for idx in range(0, len(inliers1))]
    pts2 = [list(inliers2[idx].pt) for idx in range(0, len(inliers2))]
    return pts1, pts2      



def stereoMatching(imageLeft, imageRight, method, orb_factor = 0.4):
    #Add methods code later for orb, akaze, sift
    orb = cv2.ORB_create()
    akaze = cv2.AKAZE_create()
    
    if method == 'orb':
        kpl, desl = orb.detectAndCompute(imageLeft, None)
        kpr, desr = orb.detectAndCompute(imageRight, None)
        src, dst = getCorrespondingPoints(kpl, kpr,  desl, desr, orb_factor)
    if method == 'akaze':
        kpl, desl = akaze.detectAndCompute(imageLeft, None)
        kpr, desr = akaze.detectAndCompute(imageRight, None)
        src, dst = getAkazeCorrespondingPoints(kpl, kpr,  desl, desr)
    return kpl, desl, kpr, desr, src, dst
    
def computeCommonMatches(src_base, src_cur, left_base, left_cur):
    common_base = getIntersection(src_base, left_base)
    list_base, list_cur = [], []
    for point in common_base:
        ind = left_base.index(point)
        testPoint = left_cur[ind]
        if testPoint in src_cur:
            if src_base.index(point) not in list_base and src_cur.index(testPoint) not in list_cur:
                list_base.append(src_base.index(point))
                list_cur.append(src_cur.index(testPoint))
    return list_base, list_cur

def updateIndexList(index_list, list_base_cur, list_cur):
    list_base_prev = index_list[-1]
    common_index = getIntersection(list_base_prev, list_base_cur)
    base_prev_extra = list(set(list_base_prev) - set(common_index))
    rm_index_list = []
    for i in base_prev_extra:
        rm_index_list.append(list_base_prev.index(i))
    new_index_list = []
    for i in range(0,len(index_list)):
        lst = index_list[i]
        for item in rm_index_list:
            lst[item] = -1
        lst = remove_values_from_list(lst, -1)
        new_index_list.append(lst)
    cur_updated = []
    for item in new_index_list[-1]:
        ind = list_base_cur.index(item)
        cur_updated.append(list_cur[ind])
    new_index_list.append(cur_updated)
    return new_index_list

def createPointMap(src, dst, list_base, list_cur):
    pointMap, visMap = [], []
    threeD = threeDProject(src[0], dst[0], 0)
    threeD = threeD.tolist()
    pointMap.append(threeD)
    pointMap.append(src[0])
    pointMap.append(dst[0])
    pointMap.append([[0.0, 0.0]] * len(src[0]))
    pointMap.append([[0.0, 0.0]] * len(src[0]))
    visMap.append([1] * len(src[0]))
    visMap.append([0] * len(src[0]))
    for ind1, ind2 in zip(list_base, list_cur):
        pointMap[3][ind1] = src[1][ind2]
        pointMap[4][ind1] = dst[1][ind2]
        visMap[1][ind1] = 1
    return pointMap, visMap

def updatePointMap(src, dst, list_base, list_cur, pointMap, visMap, index_list):
    threeD_0 = threeDProject(src[0], dst[0], 0)
    threeD_0 = threeD_0.tolist()
    threeD_1 = threeDProject(src[1], dst[1], 2)
    threeD_1 = threeD_1.tolist()
    pointMap[0] = threeD_0
    pointMap.append([[0.0, 0.0]] * len(src[0]))
    pointMap.append([[0.0, 0.0]] * len(src[0]))
    visMap.append([0] * len(src[0]))
    for ind1, ind2 in zip(index_list[0], index_list[2]):
        pointMap[5][ind1] = src[2][ind2]
        pointMap[6][ind1] = dst[2][ind2]
        visMap[2][ind1] = 1
    add_ind = list(set(list_base) - set(index_list[1]))
    for ind in add_ind:
        pointMap[0].append(threeD_1[ind])
        pointMap[1].append([0.0, 0.0])
        pointMap[2].append([0.0, 0.0])
        pointMap[3].append(src[1][ind])
        pointMap[4].append(dst[1][ind])
        index = list_base.index(ind)
        pointMap[5].append(src[2][list_cur[index]])
        pointMap[6].append(dst[2][list_cur[index]])
        visMap[0].append(0)
        visMap[1].append(1)
        visMap[2].append(1)
        
    return pointMap, visMap

def threeDProject(src, dst, frame):
    S = np.array(src)
    D = np.array(dst)
    S = S.reshape(1, S.shape[0], 2)
    D = D.reshape(1, D.shape[0], 2)
    #P1 = np.array([6.254584e+02, 0.000000e+00, 6.245350e+02, 0.000000e+00, 0.000000e+00, 6.254584e+02, 1.911291e+02, 0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00]).reshape(3,4)
    #P2 = np.array([6.254584e+02, 0.000000e+00, 6.245350e+02, -3.578112e+02, 0.000000e+00, 6.254584e+02, 1.911291e+02, 0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00]).reshape(3,4)
    P1 = getPMatrix(frame)
    P2 = getPMatrix(frame+1)
    ThreeDPoints = cv2.triangulatePoints(P1, P2, S, D)
    ThreeDPoints = ThreeDPoints.T
    triangulated3DPoints = np.array([x/x[3] for x in ThreeDPoints])
    return triangulated3DPoints

def calculateDistance(x1,y1,x2,y2):  
     dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)  
     return dist  
 
def getIntersection(points1, points2):
    C = [value for value in points1 if value in points2] 
    return C

def remove_values_from_list(the_list, val):
   return [value for value in the_list if value != val]

        
def removeOutliers(list_base, list_cur, src_base, src_cur, dist):
    for ite in range(0, len(list_base)):
        point1 = src_base[ite]
        point2 = src_cur[ite]
        x1,y1 = point1[0], point1[1]
        x2,y2 = point2[0], point2[1]
        d = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if d > dist:
            list_base[ite] = -1
            list_cur[ite] = -1
    list_base = remove_values_from_list(list_base, -1)
    list_cur = remove_values_from_list(list_cur, -1)
    return list_base, list_cur