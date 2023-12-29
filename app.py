import streamlit as st
import requests
from streamlit_lottie import st_lottie

from pathlib import Path
from test import *
from code import *
with st.container():
    st.title(f" Deep Learning Project website ")
    st.subheader("Cancerous CT scan detection using computer vision 🤖")
    st.write("Please enter image below 👇")
    st.write(model_results)

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
