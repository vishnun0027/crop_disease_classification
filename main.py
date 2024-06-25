import streamlit as st
from PIL import Image
from my_model import cropDiseaseModel

import json

# read json file
with open('disease.json', 'r') as f:
    DiseaseInfo = json.load(f)


predict = cropDiseaseModel()

# Sidebar title
st.sidebar.title("Upload")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"],)


with st.container( height=None, border=None):
     st.title('Crop Disease Identification')
     if uploaded_file is not None:
        image = (Image.open(uploaded_file)).resize((480, 256))
        st.image(image)

        disease = predict.predict(uploaded_file)
        st.subheader("Prediction")
        st.write(f"Predicted Disease: *{disease}*")
        st.warning(DiseaseInfo[disease])





