import cv2
import numpy as np
#import easyocr as ocr  
from PIL import Image

L = 256
#-----Function Chapter 3-----#
def Negative(imgin,imageout):
    M, N, chanel = imgin.shape
    for x in range(0, M):
        for y in range(0, N):
            r = imgin[x, y]
            s = L - 1 - r
            imageout[x, y] = s
    return imageout
def Logarit(imgin,imageout):
    print(np.max(imgin))  
    c = (255)/np.log(1+np.max(imgin))
    s = c*(np.log(1 + imgin))
    imageout = np.array(s, dtype=np.uint8)
    return imageout

def Power(imgin,imageout):
    imageout
    M, N, chanel= imgin.shape
    gamma = 5.0
    c = np.power(256 - 1, 1 - gamma)
    for x in range(0, M):
        for y in range(0, N):
            r = imgin[x, y]
            s = c*np.power(r, gamma)
            imageout[x, y] = s.astype(np.uint8)
    return imageout

def HistogramEqualization(imgin, imgout):
    M, N,ncl = imgin.shape
    h = np.zeros(L, np.int32)
    for x in range(0, M):
        for y in range(0, N):
            r = imgin[x, y]
            h[r] = h[r] + 1

    p = np.zeros(L, np.float)
    for r in range(0, L):
        p[r] = h[r]/(M*N)

    s = np.zeros(L, np.float)
    for k in range(0, L):
        for j in range(0, k + 1):
            s[k] = s[k] + p[j]
        s[k] = s[k]*(L-1)

    for x in range(0, M):
        for y in range(0, N):
            r = imgin[x, y]
            imgout[x, y] = s[r].astype(np.uint8)
    return imgout



def Smoothing(imgin):
    M, N, h = imgin.shape
    m = 21
    n = 21
    a = m // 2
    b = m // 2
    w = np.ones((m,n),np.float)/(m*n)
    imgout = cv2.filter2D(imgin,cv2.CV_8UC1, w)
    # imgout = cv2.blur(imgin, (m,n))
    return imgout

def SmoothingGauss(imgin):
    # M, N, h = imgin.shape
    # m = 51
    # n = 51
    # a = m // 2
    # b = m // 2
    # sigma = 7.0
    # w = np.zeros((m,n), np.float)
    # for s in range(-a, a+1):
    #     for t in range(-b, b+1):
    #         w[s+a, t+b] = np.exp(-(s*s + t*t)/(2*sigma*sigma))
    # sum = np.sum(w)
    # w = w/sum
    # imgout = cv2.filter2D(imgin,cv2.CV_8UC1, w)
    imgout = cv2.GaussianBlur(imgin, (5,5),cv2.BORDER_DEFAULT)
    return imgout

def MeanFilter(imgin):
    kernel = np.ones((10,10),np.float32)/25
    imgout = cv2.filter2D(imgin,-1,kernel)
    return imgout
def MedianFilter(imgin):
    imgout = cv2.medianBlur(imgin,5)
    return imgout

def Sharpen(imgin):
    w = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], np.int32)
    temp = cv2.filter2D(imgin, cv2.CV_32FC1, w)
    result = imgin - temp
    result = np.clip(result, 0, L-1)
    imgout = result.astype(np.uint8)
    return imgout
def Bileteral(imgin):
    imgout = cv2.bilateralFilter(imgin,60,60,60)
    return imgout
def UnSharpMasking(imgin):
    blur = cv2.GaussianBlur(imgin, (3, 3), 1.0).astype(np.float)
    mask = imgin - blur
    k = 10.0
    imgout = imgin + k*mask
    imgout = np.clip(imgout, 0, L-1).astype(np.uint8)
    return imgout
def LowPass(imgin):
    kernel = np.ones((10,10),np.float32)/25
    lp = cv2.filter2D(imgin,-1,kernel)
    lp = imgin - lp
    return lp

def Gradient(imgin):
    wx = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], np.int32)
    wy = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.int32)
    gx = cv2.filter2D(imgin, cv2.CV_32FC1, wx);
    gy = cv2.filter2D(imgin, cv2.CV_32FC1, wy);
    g = abs(gx) + abs(gy)
    imgout = np.clip(g, 0, L-1).astype(np.uint8)
    return imgout
