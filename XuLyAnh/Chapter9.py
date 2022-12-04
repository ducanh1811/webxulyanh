import cv2
import numpy as np

L = 256

def Erosion(imgin, imgout):
    w = cv2.getStructuringElement(cv2.MORPH_RECT,(45,45))
    cv2.erode(imgin,w,imgout)
    return imgout

def Dilation(imgin, imgout):
    w = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    cv2.dilate(imgin,w,imgout)
    return imgout
def OpeningClosing(imgin, imgout):
    w = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    temp = cv2.morphologyEx(imgin, cv2.MORPH_OPEN, w)
    cv2.morphologyEx(temp, cv2.MORPH_CLOSE, w, imgout)
    return imgout
def Boundary(imgin):
    w = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    temp = cv2.erode(imgin,w)
    imgout = imgin - temp
    return imgout

