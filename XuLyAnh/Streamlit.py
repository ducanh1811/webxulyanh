import cv2
import streamlit as st
import numpy as np
import Chapter3 as c3
import Chapter4 as c4
import Chapter9 as c9
from PIL import Image
from streamlit_option_menu import option_menu
import os
import sys
import joblib
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.utils import label_map_util   
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
background: linear-gradient(white, 	#F9B7FF);

ackground-size: 100%;
background-position: center;
background-repeat: no-repeat;
background-attachment: fixed;
color:black;
font-size:24px;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

config = tf.compat.v1.ConfigProto(gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8))
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)


CONFIG_PATH = 'Tensorflow/workspace/models_TTD/my_ssd_mobnet/pipeline.config'

CHECKPOINT_PATH = 'Tensorflow/workspace/models_TTD/my_ssd_mobnet/'

ANNOTATION_PATH = 'Tensorflow/workspace/annotations'

configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(CHECKPOINT_PATH, 'ckpt-6')).expect_partial()

def brighten_image(image, amount):
    img_bright = cv2.convertScaleAbs(image, beta=amount)
    return img_bright


def blur_image(image, amount):
    blur_img = cv2.GaussianBlur(image, (11, 11), amount)
    return blur_img


def enhance_details(img):
    hdr = cv2.detailEnhance(img, sigma_s=12, sigma_r=0.15)
    return hdr


def EditImage_loop():
    st.title("OpenCV Demo App")
    st.subheader("This app allows you to play with Image filters!")
    st.text("We use OpenCV and Streamlit for this demo")
    
    st.sidebar.title("Edit Area")
    blur_rate = st.sidebar.slider("Blurring", min_value=0.5, max_value=3.5)
    brightness_amount = st.sidebar.slider("Brightness", min_value=-50, max_value=50, value=0)
    apply_enhancement_filter = st.sidebar.checkbox('Enhance Details')
      
    st.sidebar.text("Smoothing Image")
    apply_Smoothing_filter= st.sidebar.checkbox("Smoothing")
    apply_Gauss_filter= st.sidebar.checkbox("Smoothing Gauss")
    apply_Mean_filter= st.sidebar.checkbox("Mean Filter")
    apply_Median_filter= st.sidebar.checkbox("Median Filter")
  

    st.sidebar.text("Sharpening")
    apply_Sharpen_filter= st.sidebar.checkbox("Sharpen")
    apply_UnSharpMasking_filter= st.sidebar.checkbox("UnSharpMasking")
    apply_Bileteral_filter= st.sidebar.checkbox("Bileteral")

    st.sidebar.text("Get Bound")
    apply_Boundary_filter= st.sidebar.checkbox("Boundary")
    apply_LowPass_filter= st.sidebar.checkbox("Low Pass")

    st.sidebar.text("Others")
    apply_Gradient_filter= st.sidebar.checkbox("Gradient")
    apply_Erosion_filter= st.sidebar.checkbox("Erosion")
    apply_Dilation_filter= st.sidebar.checkbox("Dilation")
    apply_OpeningClosing_filter= st.sidebar.checkbox("OpeningClosing")
    apply_Negative_filter= st.sidebar.checkbox("Negative")
    apply_Power_filter= st.sidebar.checkbox("Power")
    apply_HistogramEqualization_filter= st.sidebar.checkbox("HistogramEqualization")

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
        processed_image = c3.Negative(original_image,processed_image)
    if apply_Logarit_filter:
        processed_image = c3.Logarit(original_image,processed_image)
    if apply_Power_filter:
        processed_image = c3.Power(original_image,processed_image)
    
    if apply_HistogramEqualization_filter:
        processed_image = c3.HistogramEqualization(original_image,processed_image)
    
    if apply_Smoothing_filter:
        processed_image = c3.Smoothing(original_image) 
    if apply_Gauss_filter:
        processed_image = c3.SmoothingGauss(original_image)
    
    if apply_Sharpen_filter:
        processed_image = c3.Sharpen(original_image)
    if apply_UnSharpMasking_filter:
        processed_image = c3.UnSharpMasking(original_image)
    
    if apply_Gradient_filter: #O
        processed_image = c3.Gradient(original_image)

    if apply_Erosion_filter: #O
        processed_image = c9.Erosion(original_image,processed_image)
    if apply_Dilation_filter: #O
        processed_image = c9.Dilation(original_image,processed_image)
    if apply_OpeningClosing_filter: #O
        processed_image = c9.OpeningClosing(original_image,processed_image)
    if apply_Boundary_filter: #O
        processed_image = c9.Boundary(original_image)
    if apply_Mean_filter:
        processed_image = c3.MeanFilter(original_image)

    if apply_Median_filter: 
        processed_image = c3.MedianFilter(original_image)

    if apply_Bileteral_filter:
        processed_image = c3.Bileteral(original_image)

    if apply_LowPass_filter:
        processed_image = c3.LowPass(original_image)

    original_image = cv2.resize(original_image,(512,512))
    processed_image = cv2.resize(processed_image,(512,512))
    st.text("Original Image vs Processed Image")
    st.image([original_image, processed_image])
    st.button(label=None, key=None, help=None, on_click=processed_image.save('file.jpg'), args=None, kwargs=None, type="secondary", disabled=False)

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections
category_index = label_map_util.create_category_index_from_labelmap(ANNOTATION_PATH+'/label_map.pbtxt')

def XoaTrung(a, L):
    index = []
    flag = np.zeros(L, np.bool)
    for i in range(0, L):
        if flag[i] == False:
            flag[i] = True
            x1 = (a[i,0] + a[i,2])/2
            y1 = (a[i,1] + a[i,3])/2
            for j in range(i+1, L):
                x2 = (a[j,0] + a[j,2])/2
                y2 = (a[j,1] + a[j,3])/2
                d = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                if d < 0.2:
                    flag[j] = True
            index.append(i)
    for i in range(0, L):
        if i not in index:
            flag[i] = False
    return flag    
def DetectFruit(): 
    image_file = st.file_uploader("Upload Your Image", type=['jpg', 'png', 'jpeg'])
    if not image_file:
        return None

    imgin = np.array(Image.open(image_file))
    #r, g, b = cv2.split(imgin)
    #imgin=cv2.merge([b, g, r])
    image_np = np.array(imgin)
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)
    
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
              for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    my_box = detections['detection_boxes']
    my_class = detections['detection_classes']+label_id_offset
    my_score = detections['detection_scores']

    my_score = my_score[my_score >= 0.7]
    L = len(my_score)
    my_box = my_box[0:L]
    my_class = my_class[0:L]
        
    flagTrung = XoaTrung(my_box, L)
    my_box = my_box[flagTrung]
    my_class = my_class[flagTrung]
    my_score = my_score[flagTrung]

    # viz_utils.visualize_boxes_and_labels_on_image_array(
    #         image_np_with_detections,
    #         detections['detection_boxes'],
    #         detections['detection_classes']+label_id_offset,
    #         detections['detection_scores'],
    #         category_index,
    #         use_normalized_coordinates=True,
    #         max_boxes_to_draw=5,
    #         min_score_thresh=.5,
    #         agnostic_mode=False)

    viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            my_box,
            my_class,
            my_score,
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=5,
            min_score_thresh=.7,
            agnostic_mode=False)
    st.image(image_np_with_detections)

def DetectFace_loop():
    
    detector = cv2.FaceDetectorYN.create(
    "./face_detection_yunet_2022mar.onnx",
    "",
    (320, 320),
    0.9,
    0.3,
    5000
    )
    detector.setInputSize((320, 320))

    recognizer = cv2.FaceRecognizerSF.create(
            "./face_recognition_sface_2021dec.onnx","")

    svc = joblib.load('svc.pkl')
    mydict =   ['BanNinh', 'BanThanh','ThayDuc'
            ]
    image_file=st.file_uploader("Thêm ảnh vào", accept_multiple_files=False, key=None, help=None, on_change=None, args=None, kwargs=None,  disabled=False, label_visibility="visible")
    if not image_file:
        return None
    
    imgin = np.array(Image.open(image_file))
        #r, g, b = cv2.split(imgin)
        #imgin=cv2.merge([b, g, r])
        
    cv2.namedWindow("ImageIn", cv2.WINDOW_AUTOSIZE)
    imgin = cv2.resize(imgin,(320,320),interpolation =cv2.INTER_AREA)
    faces = detector.detect(imgin) 
    try:
        face_align = recognizer.alignCrop(imgin, faces[1][0])
        face_feature = recognizer.feature(face_align)
        test_prediction = svc.predict(face_feature)

        result = mydict[test_prediction[0]]
        st.text("Bạn này là:"+ result)
    except:
        st.text("Không nhận diện được khuôn mặt")

#color = st.color_picker('Pick A Color', '#fff1ac')
#st.write('The current color is', color )

picture = st.camera_input("Take a picture")

if picture:
    st.image(picture)
    
st.markdown(f'<h1 style="color:#33ff33;font-size:24px;">{"ColorMeBlue text”"}</h1>', unsafe_allow_html=True)


selected = option_menu("OPTION", ["Edit Image", 'Facial Recognition', "Fruit Identification"], 
        icons=["pen", "book", "envelope"], menu_icon="cast", default_index=0, orientation="horizontal")
selected

if selected == "Edit Image":
    EditImage_loop()
if selected == "Facial Recognition":
    DetectFace_loop()  
if selected == "Fruit Identification":
    DetectFruit()
