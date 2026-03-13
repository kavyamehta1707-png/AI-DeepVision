import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import streamlit as st
from torchvision import models, transforms

device = torch.device("cpu")

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Crowd Counting Dashboard",
    layout="wide"
)

st.title("AI Crowd Density Monitoring Dashboard")

# ---------------- MODEL ----------------
class CSRNet(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=None)
        self.frontend = nn.Sequential(*list(vgg.features.children())[:23])
        self.backend = nn.Sequential(
            nn.Conv2d(512,512,3,padding=2,dilation=2), nn.ReLU(),
            nn.Conv2d(512,512,3,padding=2,dilation=2), nn.ReLU(),
            nn.Conv2d(512,512,3,padding=2,dilation=2), nn.ReLU(),
            nn.Conv2d(512,256,3,padding=2,dilation=2), nn.ReLU(),
            nn.Conv2d(256,128,3,padding=2,dilation=2), nn.ReLU(),
            nn.Conv2d(128,64,3,padding=2,dilation=2), nn.ReLU(),
        )
        self.output_layer = nn.Conv2d(64,1,1)

    def forward(self,x):
        x=self.frontend(x)
        x=self.backend(x)
        x=self.output_layer(x)
        return x

@st.cache_resource
def load_model():
    model = CSRNet().to(device)
    model.load_state_dict(torch.load("model_5.pth",map_location=device))
    model.eval()
    return model

model = load_model()

# ---------------- TRANSFORM ----------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# ---------------- VIDEO PROCESSING ----------------
def process_video(video_path):

    cap=cv2.VideoCapture(video_path)

    fps=cap.get(cv2.CAP_PROP_FPS)
    if fps==0:
        fps=25

    step=max(1,int(fps*2))

    frame_count=0
    processed=0
    total_count=0
    heatmap=None

    while cap.isOpened() and processed<10:

        ret,frame=cap.read()
        if not ret:
            break

        frame_count+=1
        if frame_count%step!=0:
            continue

        processed+=1

        h,w=frame.shape[:2]

        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        img=transform(rgb).unsqueeze(0).to(device)

        with torch.no_grad():
            density=torch.relu(model(img))

        count=int(density.sum().item())
        total_count+=count

        if heatmap is None:

            density_map=density.squeeze().cpu().numpy()
            density_map=cv2.resize(density_map,(w,h))

            heatmap=cv2.normalize(density_map,None,0,255,cv2.NORM_MINMAX)
            heatmap=cv2.applyColorMap(
                heatmap.astype(np.uint8),cv2.COLORMAP_JET
            )

            heatmap=cv2.addWeighted(frame,0.6,heatmap,0.4,0)

    cap.release()

    avg_count=total_count//max(1,processed)

    return avg_count,heatmap

# ---------------- SIDEBAR ----------------
st.sidebar.header("Upload Video")

uploaded_file = st.sidebar.file_uploader(
    "Upload Crowd Video",
    type=["mp4","avi","mov"]
)

# ---------------- DASHBOARD ----------------
if uploaded_file:

    temp_path="temp_video.mp4"

    with open(temp_path,"wb") as f:
        f.write(uploaded_file.read())

    with st.spinner("Analyzing crowd density..."):
        count,heatmap=process_video(temp_path)

    col1,col2=st.columns(2)

    with col1:
        st.subheader("Crowd Heatmap")
        st.image(heatmap,channels="BGR")

    with col2:
        st.subheader("Crowd Statistics")

        st.metric(
            label="Estimated Crowd Count",
            value=count
        )

        if count<50:
            st.success("Low Crowd Density")
        elif count<150:
            st.warning("Medium Crowd Density")
        else:
            st.error("High Crowd Density")

else:
    st.info("Upload a video to start analysis.")
