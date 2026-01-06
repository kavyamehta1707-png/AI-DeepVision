import os
import cv2
import numpy as np
import tempfile
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms

# -----------------------
# DEVICE (Streamlit Cloud = CPU)
# -----------------------
device = torch.device("cpu")

# -----------------------
# CSRNet MODEL
# -----------------------
class CSRNet(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=None)
        self.frontend = nn.Sequential(*list(vgg.features.children())[:23])
        self.backend = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
        )
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x

# -----------------------
# LOAD MODEL
# -----------------------
@st.cache_resource
def load_model():
    model = CSRNet().to(device)
    
    # Path logic: Based on your GitHub 'crowd_streamlit/model_5.pth'
    # If app.py is INSIDE 'crowd_streamlit', use "model_5.pth"
    # If app.py is in the ROOT, use "crowd_streamlit/model_5.pth"
    MODEL_PATH = "model_5.pth" 
    
    if not os.path.exists(MODEL_PATH):
        # Fallback check for GitHub folder structure
        MODEL_PATH = os.path.join("crowd_streamlit", "model_5.pth")

    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found! Searched: model_5.pth and crowd_streamlit/model_5.pth")
        st.stop()

    try:
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
    except Exception as e:
        st.error(f"Error loading weights: {e}")
        st.stop()
        
    return model

model = load_model()

# -----------------------
# TRANSFORM
# -----------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# -----------------------
# UI
# -----------------------
st.title("👥 CSRNet Crowd Counting System")
option = st.radio("Select input type:", ["Image", "Video"])

# -----------------------
# IMAGE PROCESSING
# -----------------------
if option == "Image":
    img_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    if img_file:
        file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        h, w = image.shape[:2]

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_tensor = transform(rgb).unsqueeze(0).to(device)

        with torch.no_grad():
            density = torch.relu(model(img_tensor))

        count = int(density.sum().item())
        
        # Heatmap Generation
        density_map = density.squeeze().cpu().numpy()
        density_map = cv2.resize(density_map, (w, h))
        norm_map = cv2.normalize(density_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heatmap = cv2.applyColorMap(norm_map, cv2.COLORMAP_JET)

        overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
        st.image(overlay, channels="BGR", caption=f"Estimated Count: {count}")

        if count > 100:
            st.warning(f"🚨 ALERT! Large Crowd Detected: {count}")
        else:
            st.success(f"✅ Safe Crowd Count: {count}")

# -----------------------
# VIDEO PROCESSING
# -----------------------
if option == "Video":
    vid_file = st.file_uploader("Upload Video", type=["mp4", "avi"])
    if vid_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
            tfile.write(vid_file.read())
            temp_path = tfile.name

        cap = cv2.VideoCapture(temp_path)
        st.info("Processing select frames...")
        
        # Streamlit doesn't handle live video loops perfectly, so we show frames
        frame_placeholder = st.empty()
        
        processed = 0
        while cap.isOpened() and processed < 15: # Limit for demo speed
            ret, frame = cap.read()
            if not ret: break
            
            # Skip frames to speed up processing
            for _ in range(10): cap.grab() 

            processed += 1
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_tensor = transform(rgb).unsqueeze(0).to(device)

            with torch.no_grad():
                density = torch.relu(model(img_tensor))
            
            count = int(density.sum().item())
            density_map = cv2.resize(density.squeeze().cpu().numpy(), (w, h))
            heatmap = cv2.applyColorMap(
                cv2.normalize(density_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
                cv2.COLORMAP_JET
            )
            overlay = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)
            
            frame_placeholder.image(overlay, channels="BGR", caption=f"Processing Frame... Count: {count}")

        cap.release()
        os.remove(temp_path)
