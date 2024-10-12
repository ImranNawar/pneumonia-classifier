import streamlit as st
import torch
from torchvision import models
from PIL import Image
from util import classify, set_background


st.title('Chest X-Ray Pneumonia Detector')

st.header('Please upload a chest X-ray image.')

file = st.file_uploader('-', type=['jpeg', 'jpg', 'png'])

# load classifier
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=False)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 2)  # binary classification
model.load_state_dict(torch.load('../model/resnet18.pth', map_location=device))
model.to(device)
model.eval()

# load class names
# class_names = ['Normal', 'Pneumonia']
with open('../model/labels.txt', 'r') as f:
    class_names = [a[:-1].split(' ')[1] for a in f.readlines()]
    f.close()

# display image
if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)

    # classify image
    class_name, conf_score = classify(image, model, class_names)

    # write classification
    st.write("## {}".format(class_name))
    st.write("### Confidence: {:.2f}%".format(conf_score * 100))

set_background('../content/background.jpg')
# Footer
footer = """
<div style="position: fixed; bottom: 0; width: 100%; background-color: #EDF3FA; padding: 10px; text-align: center;">
      Created by Imran Nawar
</div>
"""
st.markdown(footer, unsafe_allow_html=True)