import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import io

# Define image transformations 
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load Classification Model
classification_model = models.resnext50_32x4d()
classification_model.fc = nn.Sequential(
    nn.Linear(classification_model.fc.in_features, 128),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(128, 2),  # 2 classes: Normal, Varicose
    nn.Softmax(dim=1)
)
classification_model.load_state_dict(torch.load("C:/Users/Vinod/Downloads/Varicose Vein Detection/major/varicose_resnext50.pth", map_location=torch.device('cpu')))
classification_model.eval()

# Load Staging Model
staging_model = models.resnext50_32x4d()
staging_model.fc = nn.Sequential(
    nn.Linear(2048, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 4),  # 4 stages
    nn.Softmax(dim=1)
)
staging_model.load_state_dict(torch.load("C:/Users/Vinod/Downloads/Varicose Vein Detection/major/varicose_staging_resnext50.pth", map_location=torch.device('cpu')))
staging_model.eval()

st.title("Varicose Veins Detection and Staging")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = classification_model(img_tensor)
    
    class_names = ["Normal", "Varicose"]
    pred_class = class_names[torch.argmax(output).item()]
    
    st.write(f"### Classification Result: {pred_class}")
    
    if pred_class == "Varicose":
        with torch.no_grad():
            stage_output = staging_model(img_tensor)
        stage_class = ["Stage 1", "Stage 2", "Stage 3", "Stage 4"]
        pred_stage = stage_class[torch.argmax(stage_output).item()]
        st.write(f"### Staging Result: {pred_stage}")

