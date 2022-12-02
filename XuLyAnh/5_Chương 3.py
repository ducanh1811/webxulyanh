import streamlit as st
import easyocr as ocr  
from PIL import Image
import cv2
import numpy as np

L = 255
#-----Function Chapter 3-----#
def Negative(imgin):
    global imageout
    M, N, chanel = imgin.shape
    for x in range(0, M):
        for y in range(0, N):
            r = imgin[x, y]
            s = L - 1 - r
            imageout[x, y] = s

def Logarit(imgin):
    global imageout
    c = (255)/np.log(1+255)
    imageout = c*np.log(1.0 + imgin)

    imageout = np.array(imageout, dtype=np.uint8)

def Power(imgin, gamma):
    global imageout
    M, N, chanel= imgin.shape
    gamma = 5.0
    c = np.power(256 - 1, 1 - gamma)
    for x in range(0, M):
        for y in range(0, N):
            r = imgin[x, y]
            s = c*np.power(r, gamma)
            imageout[x, y] = s.astype(np.uint8)

def PiecewiseLinear(imgin):
    global imageout
    M, N, c = imgin.shape
    rmin, rmax = cv2.minMaxLoc(imgin)
    r1 = rmin
    if rmin == 0:
        r1 = 1
    s1 = 0
    r2 = rmax
    if rmax == L-1:
        r2 = L - 2
    s2 = L - 1
    for x in range(0, M):
        for y in range(0, N):
            r = imgin[x, y]
            if r < r1:
                s = s1/r1*r
            elif r < r2:
                s = (s2-s1)/(r2-r1)*(r-r1) + s1
            else:
                s = (L-1-s2)/(L-1-r2)*(r-r2) + s2
            imageout[x, y] = s.astype(np.uint8)

def LocalHistogram(imgin):
    global imageout
    M, N, h = imgin.shape
    m = 3
    n = 3
    a = m // 2
    b = n // 2
    w = np.zeros((m,n), np.uint8)
    for x in range(a, M-a):
        for y in range(b, N-b):
            for s in range(-a, a+1):
                for t in range(-b, b+1):
                    w[s+a, t+b] = imgin[x+s, y+t]
            cv2.equalizeHist(w, w)
            imageout[x, y] = w[a, b]

def UnSharpMasking(imgin):
    blur = cv2.GaussianBlur(imgin, (3, 3), 1.0).astype(np.float)
    mask = imgin - blur
    k = 10.0
    imgout = imgin + k*mask
    imgout = np.clip(imgout, 0, L-1).astype(np.uint8)
    return imgout
def SmoothingGauss(imgin):
    M, N ,h= imgin.shape
    m = 51
    n = 51
    a = m // 2
    b = m // 2
    sigma = 7.0
    w = np.zeros((m,n), np.float)
    for s in range(-a, a+1):
        for t in range(-b, b+1):
            w[s+a, t+b] = np.exp(-(s*s + t*t)/(2*sigma*sigma))
    sum = np.sum(w)
    w = w/sum
    imgout = cv2.filter2D(imgin,cv2.CV_8UC1, w)
    # imgout = cv2.GaussianBlur(imgin, (m,n), 7.0)
    return imgout

def Smoothing(imgin):
    M, N,h = imgin.shape
    m = 21
    n = 21
    a = m // 2
    b = m // 2
    w = np.ones((m,n),np.float)/(m*n)
    imgout = cv2.filter2D(imgin,cv2.CV_8UC1, w)
    # imgout = cv2.blur(imgin, (m,n))
    return imgout

def PiecewiseLinear(imgin):
    M, N,cnl = imgin.shape
    rmin, rmax, rminloc, rmaxloc = cv2.minMaxLoc(imgin)
    r1 = rmin
    if (rmin == 0).any():
        r1 = 1
    s1 = 0
    r2 = rmax
    if rmax == L-1:
        r2 = L - 2
    s2 = L - 1
    for x in range(0, M):
        for y in range(0, N):
            r = imgin[x, y]
            if r < r1:
                s = s1/r1*r
            elif r < r2:
                s = (s2-s1)/(r2-r1)*(r-r1) + s1
            else:
                s = (L-1-s2)/(L-1-r2)*(r-r2) + s2
            imageout[x, y] = s.astype(np.uint8)

image = st.file_uploader(label = "Upload your image here",type=['png','jpg','jpeg'])

if image is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 0)
    opencv_image = cv2.cvtColor(opencv_image, cv2.IMREAD_GRAYSCALE)
    resized = cv2.resize(opencv_image,(224,224))
    # Now do something with the image! For example, let's display it:
    st.image(opencv_image)
    
    imageout=opencv_image


menu = st.sidebar.title("MENU")
with st.sidebar:
    col1, col2 = st.columns(2)
    with col1:
        naga = st.button("Nagative")
    with col2:
        Loga = st.button("Logarit")

    col3, col4 = st.columns(2)
    with col3:
        Smoot = st.button("Smoothing")
    with col4:
        Powe = st.button("Power")

    col5, col6 = st.columns(2)
    with col5:
        SmootG = st.button("Smoothing Gauss")
    with col6:
        UnShar = st.button("UnSharp Masking")
    
    Piec = st.button("PiecewiseLinear")

if (naga == True):
    st.write("Kết quả: ")
    Negative(opencv_image)
    st.image(imageout)

#Loga = st.sidebar.button("Logarit")
if (Loga == True):
    Logarit(opencv_image)
    st.image(imageout)

#Powe = st.sidebar.button("Power")
if(Powe == True):
    Power(opencv_image, 1)
    st.image(imageout)

#UnShar = st.sidebar.button("UnSharpMasking")
if(UnShar == True):
    imaout= UnSharpMasking(opencv_image)
    st.image(imaout)

#SmootG = st.sidebar.button("SmoothingGauss")
if(SmootG == True):
    imaout = SmoothingGauss(opencv_image)
    st.image(imaout)

#Smoot = st.sidebar.button("Smoothing")
if(Smoot == True):
    imaout = Smoothing(opencv_image)
    st.image(imaout)

if Piec:
    PiecewiseLinear(opencv_image)
    st.image(imageout)