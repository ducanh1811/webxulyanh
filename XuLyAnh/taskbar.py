import streamlit as st
from streamlit_option_menu import option_menu
import cv2
import numpy as np
import Chapter3 as c3
import Chapter4 as c4
import Chapter9 as c9
from PIL import Image

def blur_image(image, amount):
    blur_img = cv2.GaussianBlur(image, (11, 11), amount)
    return blur_img
def brighten_image(image, amount):
    img_bright = cv2.convertScaleAbs(image, beta=amount)
    return img_bright
def enhance_details(img):
    hdr = cv2.detailEnhance(img, sigma_s=12, sigma_r=0.15)
    return hdr
selected = option_menu(
    menu_title=None,  # required
    options=["Home", "Projects", "Contact"],  # required
    icons=["house", "book", "envelope"],  # optional
    menu_icon="cast",  # optional
    default_index=0,  # optional
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#fafafa"},
        "icon": {"color": "orange", "font-size": "25px"},
        "nav-link": {
            "font-size": "25px",
            "text-align": "left",
            "margin": "0px",
            "--hover-color": "#eee",
        },
        "nav-link-selected": {"background-color": "green"},
        },
)

def main_loop():
    st.title("OpenCV Demo App")
    st.subheader("This app allows you to play with Image filters!")
    st.text("We use OpenCV and Streamlit for this demo")
    #with st.sidebar: 
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
if selected == "Home":
    main_loop()
if selected == "Projects":
    st.title(f"You have selected {selected}")
if selected == "Contact":
    st.title(f"You have selected {selected}")