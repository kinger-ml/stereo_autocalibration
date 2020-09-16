# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 22:57:03 2020

@author: Krishna Kinger
"""
import numpy as np
import cv2
import json

def read3dpoints(filename):
    file = open(filename, 'r')
    str=''
    #skip first two lines
    file.readline()
    file.readline()
    str+=file.readline()
    str = str.replace('mat: [ ','')
    str+=file.read()
    str = str.replace(']','')
    str = str.replace(',',' ')
    threeD = str.split()
    #[float(i) for i in threeD]
    threeD = np.reshape(np.array(threeD, dtype=np.float32), (int(len(threeD)/3),3))
    file.close()
    return threeD


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
    R, _ = cv2.Rodrigues(rotation)
    return R

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

def remove_values_from_list(the_list, val):
   return [value for value in the_list if value != val]

def writePointMap(pointMap, visMap, baseframe, curframe):
    WP = []
    for point in pointMap[0]:
        point.pop()
        WP.append(point)
    #Write 3D points to file
    with open('data/camera/threeD.txt', 'w') as filehandle:
        json.dump(WP, filehandle)
    #Remove non visible indexes from images before writing to file
    for cam in range(0, len(pointMap)-1):
        cam_points = remove_values_from_list(pointMap[cam+1], [0.0, 0.0])
        cam = 'data/camera/cam'+str(2*baseframe+cam)+'.txt'
        with open(cam, 'w') as filehandle:
            json.dump(cam_points, filehandle)
            
    #Write Visibility matrix to file
    for i in range(0, len(visMap)):
        vis = 'data/camera/visibility'+str(2*(baseframe+i))+'.txt'
        with open(vis, 'w') as filehandle:
            json.dump(visMap[i], filehandle)
            
    #Write Configuration File
    cfg = str(2 * len(visMap)) + " " + str(len(pointMap[0])) + " " + str(baseframe) + " " + str(curframe)
    with open('data/camera/sba_cfg.txt', 'w') as filehandle:
        json.dump(cfg, filehandle)
        
def writeToFile(threeD, src_base, dst_base, src_cur, dst_cur, list_base, list_cur, frame):
    WP, S, D, visMap = [], [], [], []
    visMap.append([1] * len(src_base))
    visMap.append([0] * len(src_base))
    for point in threeD:
        point.pop()
        WP.append(point)
    #Write 3D points to file
    with open('data/camera/threeD.txt', 'w') as filehandle:
        json.dump(WP, filehandle)
    for ind in list_cur:
        S.append(src_cur[ind])
        D.append(dst_cur[ind])
    for ind in list_base:
        visMap[1][ind] = 1    
    #Write Cam Points
    cam_i = 2*frame
    cam = 'data/camera/cam'+str(cam_i)+'.txt'
    with open(cam, 'w') as filehandle:
        json.dump(src_base, filehandle)
    cam = 'data/camera/cam'+str(cam_i+1)+'.txt'
    with open(cam, 'w') as filehandle:
        json.dump(dst_base, filehandle)
    cam = 'data/camera/cam'+str(cam_i+2)+'.txt'
    with open(cam, 'w') as filehandle:
        json.dump(S, filehandle)
    cam = 'data/camera/cam'+str(cam_i+3)+'.txt'
    with open(cam, 'w') as filehandle:
        json.dump(D, filehandle)
        
    #Write Visibility matrix to file
    vis = 'data/camera/visibility'+str(cam_i)+'.txt'
    with open(vis, 'w') as filehandle:
        json.dump(visMap[0], filehandle)
    vis = 'data/camera/visibility'+str(cam_i+2)+'.txt'
    with open(vis, 'w') as filehandle:
        json.dump(visMap[1], filehandle)
            
    #Write Configuration File
    cfg = str(2 * len(visMap)) + " " + str(len(visMap[0])) + " " + str(frame)
    with open('data/camera/sba_cfg.txt', 'w') as filehandle:
        json.dump(cfg, filehandle)

def writeList(lst, file):
    with open(file, 'w') as filehandle:
        json.dump(lst, filehandle)
    

def saveResults(src,dst, index_list, num):
    for ite in range(0,len(index_list)):
        s = "{:06d}".format(ite+num)
        im = 'data/2010_03_17_drive_0046/I1_' + s + '.png'
        img = cv2.imread(im)
        for ind in index_list[ite]:
            point = src[ite][ind]
            point = [int(i) for i in point]
            cv2.circle(img, tuple(point), 2, (0, 0, 255), -1) 
        res = 'data/results2/' + str(ite+num) + '.png'
        cv2.imwrite(res, img)
        

    
    