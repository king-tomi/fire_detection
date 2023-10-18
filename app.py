import streamlit as st
import PIL
import numpy as np
import joblib
from ultralytics import YOLO
import os
import pandas as pd


YOLO_MODEL_PATH = os.path.join(os.getcwd(), 'best.pt')
SENSOR_MODEL_PATH = os.path.join(os.getcwd(), 'model.pkl')
TEST_PATH = os.path.join(os.getcwd(), 'test.csv')
TEST_LABEL_PATH = os.path.join(os.getcwd(), 'test_labels.csv')
IMG_PATH = os.path.join(os.getcwd(), 'test/')

# Setting page layout
st.set_page_config(
    page_title="Fire Detection For Forests",  # Setting page title
    page_icon="🤖",     # Setting page icon
    layout="wide",      # Setting layout to wide
    initial_sidebar_state="expanded"    # Expanding sidebar by default
)

# Creating sidebar
with st.sidebar:
    st.header("Image/Video Config")     # Adding header to sidebar

    # Adding file uploader to sidebar for selecting images
    source_img = st.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

# Creating main page heading
st.title("Fire Detection For Forests")

# Creating two columns on the main page
col1, col2 = st.columns(2)

# Adding image to the first column if image is uploaded
with col1:
    if source_img:
        # Opening the uploaded image
        if "non" in source_img.name:
            img_path = IMG_PATH + "non_fire/" + source_img.name
            uploaded_image = PIL.Image.open(IMG_PATH + "non_fire/" + source_img.name)
            #selecting a sample sensor data
            sensor = pd.read_csv(TEST_PATH).sample(n=1)
            # Adding the uploaded image to the page with a caption
            st.image(source_img,
                    caption="Uploaded Image",
                    use_column_width=True
                    )
            st.dataframe(sensor)
        else:
            img_path = IMG_PATH + "fire/" + source_img.name
            uploaded_image = PIL.Image.open(IMG_PATH + "fire/" + source_img.name)
            #selecting a sample sensor data
            sensor = pd.read_csv(TEST_PATH).sample(n=1)
            # Adding the uploaded image to the page with a caption
            st.image(source_img,
                    caption="Uploaded Image",
                    use_column_width=True
                    )
            st.dataframe(sensor)

try:
    model = YOLO(YOLO_MODEL_PATH)
    sensor_model = joblib.load(SENSOR_MODEL_PATH)
    print('Models loaded successfuly')
except Exception as e:
    st.error(
        f"Unable to load models. Check the specified path: {YOLO_MODEL_PATH} and {SENSOR_MODEL_PATH}")
    st.error(e)

if st.sidebar.button('Detect Fires'):
    res = model.predict(img_path)
    sensor_res = sensor_model.predict(sensor)[0]
    names = res[0].names
    clas = names[res[0].probs.top1]
    boxes = res[0].boxes
    res_plotted = res[0].plot()[:, :, ::-1]

    with col2:
        if (clas, sensor_res) == ('fire', 1):
            st.write('A fire is Occurring!!! Please take appropriate measures NOW')
        elif (clas, sensor_res) == ('fire', 0):
            st.write('A fire is likely to occur soon, please take appropriate measures')
        elif (clas, sensor_res) == ('non_fire', 1):
            st.write('A fire is likely to occur, Please take appropriate measures')
        elif (clas, sensor_res) == ('non_fire', 0):
            st.write('There is no possibility of fire')