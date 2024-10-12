import base64
import streamlit as st
import torch
from torchvision import transforms


def set_background(image_file):
    """
    This function sets the background of a Streamlit app to an image specified by the given image file.

    Parameters:
        image_file (str): The path to the image file to be used as the background.

    Returns:
        None
    """
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)


def classify(image, model, class_names):
    """
    This function takes an image, a model, and a list of class names and returns the predicted class and confidence
    score of the image.

    Parameters:
        image (PIL.Image.Image): An image to be classified.
        model (tensorflow.keras.Model): A trained machine learning model for image classification.
        class_names (list): A list of class names corresponding to the classes that the model can predict.

    Returns:
        A tuple of the predicted class name and the confidence score for that prediction.
    """
    # Preprocess the image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = transform(image).unsqueeze(0)
    
    # Move the input and model to GPU for speed if available
    device = next(model.parameters()).device
    image = image.to(device)
    
    # Perform the prediction
    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
        confidence_score = torch.nn.functional.softmax(outputs, dim=1)[0][preds[0]].item()

        # probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        
        # index = 0 if probabilities[0] > 0.95 else 1
        # confidence_score = probabilities[index].item() # typical way with threshold
    
    # Get the predicted class name
    predicted_class_name = class_names[preds[0]]
    
    return predicted_class_name, confidence_score

