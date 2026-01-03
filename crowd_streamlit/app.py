import streamlit as st
import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
import tempfile

# -----------------------
# DEVICE
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
        self.output_layer = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        return self.output_layer(self.backend(self.frontend(x)))

# -----------------------
# LOAD MODEL
# -----------------------
@st.cache_resource
def load_model():
    model = CSRNet().to(device)
    model.load_state_dict(torch.load("model_5.pth", map_location=device))
    model.eval()
    return model

model = load_model()

# -----------------------
# TRANSFORM
# -----------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------
# UI
# -----------------------
st.title("CSRNet Crowd Counting System")

option = st.radio("Select input type:", ["Image", "Video"])

# -----------------------
# IMAGE PROCESSING
# -----------------------
if option == "Image":
    img_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if img_file:
        image = np.array(bytearray(img_file.read()), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        h, w = image.shape[:2]

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = transform(rgb).unsqueeze(0).to(device)

        with torch.no_grad():
            density = torch.relu(model(img))

        count = int(density.sum().item())

        density_map = density.squeeze().cpu().numpy()
        density_map = cv2.resize(density_map, (w, h))
        heatmap = cv2.applyColorMap(
            cv2.normalize(density_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
            cv2.COLORMAP_JET
        )

        overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)

        st.image(overlay, channels="BGR", caption=f"Estimated Count: {count}")

        if count > 100:
            st.error(f"ALERT! Crowd Count: {count}")
        else:
            st.success(f"Safe Crowd Count: {count}")

# -----------------------
# VIDEO PROCESSING
# -----------------------
if option == "Video":
    vid_file = st.file_uploader("Upload Video (short)", type=["mp4", "avi"])

    if vid_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(vid_file.read())

        cap = cv2.VideoCapture(tfile.name)
        fps = cap.get(cv2.CAP_PROP_FPS)
        step = int(fps * 2)
        processed = 0
        MAX_FRAMES = 20

        st.info("Processing video...")

        while cap.isOpened() and processed < MAX_FRAMES:
            ret, frame = cap.read()
            if not ret:
                break

            if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % step != 0:
                continue

            processed += 1
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = transform(rgb).unsqueeze(0).to(device)

            with torch.no_grad():
                density = torch.relu(model(img))

            count = int(density.sum().item())

            density_map = density.squeeze().cpu().numpy()
            density_map = cv2.resize(density_map, (w, h))
            heatmap = cv2.applyColorMap(
                cv2.normalize(density_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
                cv2.COLORMAP_JET
            )

            overlay = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)

            st.image(overlay, channels="BGR", caption=f"Frame Crowd Count: {count}")

        cap.release()
