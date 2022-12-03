import os
import joblib
import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import pytesseract
#import tensorflow
import cv2
import numpy as np
import sys
import os
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
#import ThucHanhXuLyAnh as codeChapter



import tkinter
from tkinter import Frame, Tk, BOTH, Text, Menu, END
from tkinter.filedialog import Open, SaveAs


with st.sidebar:
    selected = option_menu("Ứng dụng", ["Khuôn mặt", 'Trái cây',"Đọc chữ số"], 
        icons=['person lines fill', 'apple','book'], menu_icon="arrow-bar-down", default_index=1)


if selected=="Khuôn mặt":
    st.title("Nhận diện khuôn mặt")
elif selected=='Trái cây':
    st.title("Nhận diện trái cây")
    #st.image(img)

else:
    st.title("Nhận diện phép toán")

img=st.file_uploader("Thêm ảnh vào", accept_multiple_files=False, key=None, help=None, on_change=None, args=None, kwargs=None,  disabled=False, label_visibility="visible")

if img is not None:
    st.image(img)
    
config = tf.compat.v1.ConfigProto(gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
#device_count = {'GPU': 1}
)
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

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections


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


if selected=="Khuôn mặt":
    st.balloons()
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
    mydict =   ['Anh'
            ,'Kha','Lam','Ngoc','Phat' ,'Phuc','Thai','Thang','Thanh','Vi'
            ]
    if img is not None:
        imgin = np.asarray(Image.open(img))
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
            st.text("Bạn này tên là:"+ result)
        except:
            st.text("Không nhận diện được khuôn mặt")
elif selected=='Trái cây':
    if img is not None:
        imgin = np.asarray(Image.open(img))
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
        #cv2.imshow("ImageIn",  image_np_with_detections)
elif selected=="Đọc chữ số":
    if img is not None:
        os.environ['TESSDATA_PREFIX'] = './tessdata-main'
        imgin = np.asarray(Image.open(img))
        #r, g, b = cv2.split(imgin)
        #imgin=cv2.merge([b, g, r])
        gray = cv2.cvtColor(imgin, cv2.COLOR_BGR2GRAY)
        #gray=img
        # Apply dilation and erosion to remove some noise
        kernel = np.ones((2, 3), np.uint8)
        gray = cv2.dilate(gray, kernel, iterations=1)
        gray = cv2.erode(gray, kernel, iterations=2)

        # Write image after removed noise
        cv2.imwrite("./removed_noise.png", gray)

        #  Apply threshold to get image with only black and white
        # img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
        _, gray = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # Write the image after apply opencv to do some ...
        cv2.imwrite("./thres.png", gray)

        # Recognize text with tesseract for python
        pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
        result1 = pytesseract.image_to_string(Image.open("./thres.png"))
        
        if False:
            gray=~gray
            cv2.imwrite("./thres.png", gray)

            # Recognize text with tesseract for python
            pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
            result2 =pytesseract.image_to_string(Image.open("./thres.png"))
            
        st.text("Bài toán là:"+result1)