# to run use
# streamlit run app.py

import tempfile
import cv2 as cv
import numpy as np
import streamlit as st
import torch
# from detect import detect
from PIL import Image
from io import *
import glob
from datetime import datetime
import os
import wget
import time

model = None
cfg_model_path = "carPartDetection5.pt"
Demo_img = "car1.jpeg"
Demo_video = "car_test1.mp4"


@st.experimental_singleton
def load_model(path,device):
    model_ = torch.hub.load('ultralytics/yolov5', 'custom', path=cfg_model_path, force_reload=True)
    model_.to(device)
    print("model is loaded")
    return model_

def infer_image(img):
    model.conf = confidence
    result = model(img)
    result.render()
    image = Image.fromarray(result.ims[0])

    return image

def image_input(data_src):
    img_file = None
    if data_src == "Sample Data":
        img_file = Demo_img
    else:
        img_byte = st.sidebar.file_uploader("upload an image" ,type=["png","jpeg","jpg"])
        if img_byte:
            # img_file = "data/uploads/test_img." + img_byte.name.split('.')[-1]
            img_file = os.path.join("data" , "uploads" , f"test_img.{img_byte.name.split('.')[-1]}")
            Image.open(img_byte).save(img_file)

    if img_file:
        img = infer_image(img_file)
        st.image(img , caption = "model Prediction")

def video_input(data_src):
   vid_file = None
   if data_src == "Sample Data":
       vid_file = Demo_video
   else:
       video_file_buffer = st.sidebar.file_uploader("Upload a video" , type=['mp4','mov','avi','m4v','asf'])
       if video_file_buffer:
           # vid_file = "data/uploads/test_upload." + video_file_buffer.name.split('.')[-1]
           vid_file = os.path.join("data", "uploads", f"test_upload.{video_file_buffer.name.split('.')[-1]}")
           with open(vid_file , 'wb') as out:
               out.write(video_file_buffer.read())

   if vid_file:
       cap = cv.VideoCapture(vid_file)
       width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
       height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
       st.markdown("____")
       output = st.empty()
       while True:
           ret, frame = cap.read()
           if not ret:
               print("cant receive frame")
               break
           frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
           output_img = infer_image(frame)
           output.image(output_img)
       cap.release()


def main():

    global model , confidence , cfg_model_path
    st.title("Car part Detection")

    st.sidebar.title("Settings")

    # load model
    model = load_model(cfg_model_path , device="cpu")

    # confidence
    confidence = st.sidebar.slider('Confidence',min_value=0.1, max_value=1.0 , value=0.50)

    # use custom classes
    # names = ['Fog light', 'Headlight', 'License plate', 'Side mirror', 'Tail light', 'Wheel', 'Windshield', 'bb']
    if st.sidebar.checkbox("Custom classes"):
        print(model.names)
        model_names = list(model.names.values())
        assigned_class = st.sidebar.multiselect("Select the custom class", model_names, default='Windshield')
        classes = [model_names.index(name) for name in assigned_class]
        model.classes = classes
    else:
        model.classes = list(model.names.keys())

    st.sidebar.markdown('------')

    # input options
    options = st.sidebar.radio("select input type" , ["Image" , "Video"])
    # input src options
    data_src = st.sidebar.radio("select input source" , ["Sample Data" , "upload your own data"] )

    if options == "Image":
        print("image selected")
        image_input(data_src)
    else:
        print("video selected")
        video_input(data_src)

if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        pass


