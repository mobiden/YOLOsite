import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np

#@st.cache_data

model = YOLO("https://github.com/mobiden/YOLOsite/blob/bb698daf04e869c4d6906a51b323a79c695e5bb1/3506_best.pt")

st.title('Поиск оружия на фотографиях')

#picture = st.camera_input("Take a picture")

#if picture:
#    st.image(picture)

uploaded_file = st.file_uploader(label="Загрузите кадр с оружием",
                 type=['png', 'jpg', ],
                 accept_multiple_files=False,
                 key=None,
                 help=None,
                 on_change=None,
                 args=None,
                 disabled=False,
                 label_visibility="visible")
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    result = model.predict(img)
    result = result[0].plot()
    st.image(result)
