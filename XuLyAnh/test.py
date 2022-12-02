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
    print(np.max(imgin))  
    c = (255)/np.log(1+np.max(imgin))
    s = c*(np.log(1 + imgin))
    imageout = np.array(s, dtype=np.uint8)
    return imageout

def Power(imgin):
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

def Piecewise(imgin, r1,s1,r2, s2):
    if (0 <=imgin and imgin <= r1):
        return (s1/r1)*imgin
    elif (r1 < imgin and imgin <= r2):
        return ((s2 - s1)/(r2 - r1))*(imgin - r1) + s1
    else:
        return ((255-s2)/(255-r2))*(imgin-r2) + s2


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
    naga = st.button("Nagative")
    if (naga == True):
        Negative(opencv_image)
        st.image(imageout)

    Loga = st.button("Logarit")
    if (Loga == True):
        Logarit(opencv_image)
        st.image(imageout)

    Powe = st.button("Power")
    if(Powe == True):
        Power(opencv_image)
        st.image(imageout)

    Piec = st.button("PiecewiseLinear")
    if(Piec == True):
        r1 = 70
        s1 = 0
        r2 = 140
        s2 = 255
        imageout = Piecewise(opencv_image,r1,s1,r2, s2)
        st.image(imageout)