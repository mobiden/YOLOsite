import io

import streamlit as st
from ultralytics import YOLO

from PIL import Image
#@st.cache_data

model = YOLO("3506_best.pt")

st.title('Поиск оружия на фотографиях')


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
    bytes_data = uploaded_file.read()

    img = Image.open(io.BytesIO(bytes_data))

   # st.image(img)
    result = model.predict(img)
    result = result[0].plot()
    
    im = Image.fromarray(result)  # RGB PIL image
    st.image(result, channels="BGR")
