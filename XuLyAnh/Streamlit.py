import cv2
import streamlit as st
import numpy as np
import Chapter3 as c3
import Chapter4 as c4
import Chapter9 as c9
from PIL import Image
# from streamlit_option_menu import option_menu

def brighten_image(image, amount):
    img_bright = cv2.convertScaleAbs(image, beta=amount)
    return img_bright


def blur_image(image, amount):
    blur_img = cv2.GaussianBlur(image, (11, 11), amount)
    return blur_img


def enhance_details(img):
    hdr = cv2.detailEnhance(img, sigma_s=12, sigma_r=0.15)
    return hdr


def main_loop():
    st.title("OpenCV Demo App")
    st.subheader("This app allows you to play with Image filters!")
    st.text("We use OpenCV and Streamlit for this demo")

    st.sidebar.title("Edit")
    blur_rate = st.sidebar.slider("Blurring", min_value=0.5, max_value=3.5)
    brightness_amount = st.sidebar.slider("Brightness", min_value=-50, max_value=50, value=0)
    apply_enhancement_filter = st.sidebar.checkbox('Enhance Details')
    st.sidebar.title("C3")
    apply_Negative_filter= st.sidebar.checkbox("Negative")
    apply_Logarit_filter= st.sidebar.checkbox("Logarit")
    apply_Power_filter= st.sidebar.checkbox("Power")
    apply_PiecewiseLinear_filter= st.sidebar.checkbox("PiecewiseLinear")
    apply_Histogram_filter= st.sidebar.checkbox("Histogram")
    apply_HistogramEqualization_filter= st.sidebar.checkbox("HistogramEqualization")
    apply_LocalHistogram_filter= st.sidebar.checkbox("LocalHistogram")
    apply_HistogramStatistics_filter= st.sidebar.checkbox("HistogramStatistics")
    apply_MySmoothing_filter= st.sidebar.checkbox("MySmoothing")
    apply_Smoothing_filter= st.sidebar.checkbox("Smoothing")
    apply_Gauss_filter= st.sidebar.checkbox("Gauss")
    apply_MySort_filter= st.sidebar.checkbox("MySort")
    apply_MedianFilter_filter= st.sidebar.checkbox("MedianFilter")
    apply_MySharpen_filter= st.sidebar.checkbox("MySharpen")
    apply_Sharpen_filter= st.sidebar.checkbox("Sharpen")
    apply_UnSharpMasking_filter= st.sidebar.checkbox("UnSharpMasking")
    apply_MyGradient_filter= st.sidebar.checkbox("MyGradient")
    apply_Gradient_filter= st.sidebar.checkbox("Gradient")
    
    st.sidebar.title("C4")
    apply_Spectrum_filter= st.sidebar.checkbox("Spectrum")
    apply_FrequencyFilter_filter= st.sidebar.checkbox("FrequencyFilter")
    apply_DrawFilter_filter= st.sidebar.checkbox("DrawFilter")
    apply_NotchRejectFilter_filter= st.sidebar.checkbox("NotchRejectFilter")
    apply_RemoveMoire_filter= st.sidebar.checkbox("RemoveMoire")


    st.sidebar.title("C9")
    apply_Erosion_filter= st.sidebar.checkbox("Erosion")
    apply_Dilation_filter= st.sidebar.checkbox("Dilation")
    apply_OpeningClosing_filter= st.sidebar.checkbox("OpeningClosing")
    apply_Boundary_filter= st.sidebar.checkbox("Boundary")
    apply_HoleFill_filter= st.sidebar.checkbox("HoleFill")
    apply_MyConnectedComponent_filter= st.sidebar.checkbox("MyConnectedComponent")
    apply_ConnectedComponent_filter= st.sidebar.checkbox("ConnectedComponent")
    apply_CountRice_filter= st.sidebar.checkbox("CountRice")

    image_file = st.file_uploader("Upload Your Image", type=['jpg', 'png', 'jpeg'])
    if not image_file:
        return None

    original_image = Image.open(image_file)
    original_image = np.array(original_image)
    
    processed_image = blur_image(original_image, blur_rate)
    processed_image = brighten_image(processed_image, brightness_amount)

    if apply_enhancement_filter:
        processed_image = enhance_details(processed_image)

    # C3 ----------------------------------------------------------------- 
    if apply_Negative_filter: 
        processed_image = np.zeros(original_image.shape,np.uint8)
        processed_image = c3.Negative(original_image,processed_image)
    if apply_Logarit_filter:
        processed_image = c3.Logarit(original_image,processed_image)
    if apply_Power_filter:
        processed_image = c3.Power(original_image,processed_image)
    if apply_PiecewiseLinear_filter:
        processed_image = c3.PiecewiseLinear(original_image,processed_image)
    if apply_Histogram_filter:
        processed_image = c3.Histogram(original_image,processed_image)
    if apply_HistogramEqualization_filter:
        processed_image = c3.HistogramEqualization(original_image,processed_image)
    if apply_LocalHistogram_filter:
        processed_image = c3.LocalHistogram(original_image,processed_image)
    if apply_HistogramStatistics_filter:
        processed_image = c3.HistogramStatistics(original_image, processed_image)
    if apply_MySmoothing_filter:
        processed_image = c3.MySmoothing(original_image,processed_image)
    if apply_Smoothing_filter:
        processed_image = c3.Smoothing(original_image) 
    if apply_Gauss_filter:
        processed_image = c3.SmoothingGauss(original_image)
    if apply_MySort_filter:
        processed_image = c3.MySort(original_image)
    if apply_MedianFilter_filter:
        processed_image = c3.MedianFilter(original_image,processed_image)
    if apply_MySharpen_filter:
        processed_image = c3.MySharpen(original_image,processed_image)
    if apply_Sharpen_filter:
        processed_image = c3.Sharpen(original_image)
    if apply_UnSharpMasking_filter:
        processed_image = c3.UnSharpMasking(original_image)
    if apply_MyGradient_filter: #X
        processed_image = c3.MyGradient(original_image,processed_image)
    if apply_Gradient_filter: #O
        processed_image = c3.Gradient(original_image)

    # C4 -----------------------------------------------------------
    if apply_Spectrum_filter: #O
        processed_image = c4.Spectrum(original_image)
    if apply_FrequencyFilter_filter: #O
        processed_image = c4.FrequencyFilter(original_image)
    if apply_DrawFilter_filter: #O
        processed_image = c4.DrawFilter(original_image)
    if apply_NotchRejectFilter_filter: #O
        processed_image = c4.NotchRejectFilter(original_image,processed_image)
    if apply_RemoveMoire_filter: #O
        processed_image = c4.RemoveMoire(original_image)
    # C9 -----------------------------------------------------------
    if apply_Erosion_filter: #O
        processed_image = c9.Erosion(original_image,processed_image)
    if apply_Dilation_filter: #O
        processed_image = c9.Dilation(original_image,processed_image)
    if apply_OpeningClosing_filter: #O
        processed_image = c9.OpeningClosing(original_image,processed_image)
    if apply_Boundary_filter: #O
        processed_image = c9.Boundary(original_image)
    if apply_HoleFill_filter: #O
        processed_image = c9.HoleFill(original_image)
    if apply_MyConnectedComponent_filter: #O
        processed_image = c9.MyConnectedComponent(original_image)
    if apply_ConnectedComponent_filter: #O
        processed_image = c9.ConnectedComponent(original_image)
    if apply_CountRice_filter: #O
        processed_image = c9.CountRice(original_image)
    
    original_image = cv2.resize(original_image,(512,512))
    processed_image = cv2.resize(processed_image,(512,512))
    st.text("Original Image vs Processed Image")
    st.image([original_image, processed_image])


if __name__ == '__main__':
    main_loop()
