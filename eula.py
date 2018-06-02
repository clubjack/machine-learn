# -*- coding: utf-8 -*-
"""
Created on Sat May 19 21:29:47 2018

@author: admin
"""
import cv2
import numpy as np

def get_EuclideanDistance(x,y):
    myx = np.array(x)
    myy = np.array(y)
    return np.sqrt(np.sum((myx-myy)*(myx-myy)))

def findpic(img,findimg,h,fh,w,fw):
    minds = 1e8
    mincb_h = 0
    mincb_w \ 0
    for now_h in range(0,h-fh):
        for now_w in range(0,w-fw):
            my_img = img[now_h:now_h+fh,now_w:now_w+fw,:]
            my_findimg = findimg
            dis = get_EuclideanDistance(my_img,my_findimg)
            if dis<minds:
                mincb_w = now_w
                mincb_h = now_h
                minds = dis
        print('.',)
    findpt = mincb_w,mincb_h
    cv2.rectangle(img,findpt,(findpt[0]+fw,findpt[1]+fh),(0,0,255))
    return img

def showpiclocation(img,findimg):
    w = img.shape[1]
    h = img.shape[0]
    fw = findimg.shape[1]
    fh = findimg.shape[0]
    return findpic(img,findimg,h,fh,w,fw)
    
def addnoise(img):
    countn=500000
    for k in range(0,countn):
        xi = int(np.random.uniform(0,img.shape[1]))
        yi = int(np.random.uniform(0,img.shape[0]))
        img[xj,xi,0] = 255*np.random.rand()