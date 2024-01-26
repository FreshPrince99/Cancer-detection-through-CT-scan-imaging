import streamlit as st
import requests
from streamlit_lottie import st_lottie
import torchvision
import numpy as np
from pathlib import Path

import torch
from torch import nn
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from torchvision.datasets import ImageFolder
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import torch
from torch import nn
# from model_LogR import *
with st.container():
    st.title(f" Deep Learning Project website ")
    st.subheader("Cancerous CT scan detection using computer vision 🤖")
    st.write("Please enter image below 👇")
    # st.write(model_results)

def load_lottieurl(url):
    r= requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


#ANIMATIONS
lottie_coding = load_lottieurl("https://lottie.host/481c2b9e-f3c5-480b-899e-ac0a95640927/y9qc6q26cz.json")

with st.container():
    st.write("---")
    left_column,right_column = st.columns(2)
    with left_column:
        st.write('Currently this model accepts only 1 image')
        uploaded_file = st.file_uploader(label = 'Insert image', accept_multiple_files = False,
                                        type = ['png','jpeg','jpg'])
        # Now this image will be sent to the main model file to predict and bring back the prediction as well as loss and accuracy values
        uploaded_file = uploaded_file.read()
        uploaded_file=torch.Tensor(np.frombuffer(uploaded_file, dtype=np.int32))
        convert = transforms.ToTensor()
        converted_file = uploaded_file.type(torch.float32) 
        # uploaded_file = torchvision.io.decode_jpeg(uploaded_file)
        st.write(converted_file)
        st.write(converted_file.dtype)
        st.write(converted_file.shape)

        custom_image_transform = transforms.Compose([
            transforms.Resize(size=(64, 64))
        ])

        # Transform target image
        custom_image_transformed = custom_image_transform(converted_file)
        st.write(custom_image_transformed.shape)

        
    # with left_column:
    #     st.header("A Brief summary ")
    #     st.write(
    #             """
                
    #             - This website has been created to detect cancerous ct scan images of the chest
    #             - The entire website has been created by making use of streamlit
    #             - Head below to the bottom section and upload an image and check the accuracy of the model 
    #             . Good luck!
                
    #             """
    #         )

    with right_column:
        st_lottie(lottie_coding, height =300, key="coding")
