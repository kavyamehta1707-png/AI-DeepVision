import os
import cv2
import numpy as np
import torch
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import matplotlib.pyplot as plt

image_dir = r"archive (2)\ShanghaiTech\part_B\train_data\images"
gt_dir = r"archive (2)\ShanghaiTech\part_B\train_data\ground-truth"

save_img_dir = r"processed_partB\images"
save_den_dir = r"processed_partB\density_maps"

resize_shape = (256, 256)   # width, height
sigma = 4  # Gaussian blur factor (lower = sharper heads)

# create output folders
os.makedirs(save_img_dir, exist_ok=True)
os.makedirs(save_den_dir, exist_ok=True)

# FUNCTION: Generate density map
def generate_density_map(image, points, sigma=4):
    h, w, _ = image.shape
    density = np.zeros((h, w), dtype=np.float32)

    if len(points) == 0:
        return density

    for x, y in points:
        x = int(min(w - 1, max(0, x)))
        y = int(min(h - 1, max(0, y)))
        density[y, x] = 1

    density = gaussian_filter(density, sigma=sigma)
    return density

# PROCESS ALL IMAGES
all_images = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]

print(f"Found {len(all_images)} images.")
print("Starting preprocessing...\n")

for img_name in tqdm(all_images):

    img_path = os.path.join(image_dir, img_name)
    mat_path = os.path.join(gt_dir, "GT_" + img_name.replace(".jpg", ".mat"))

    # Load image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Load ground truth points
    gt = loadmat(mat_path)
    points = gt["image_info"][0][0][0][0][0]

    # Resize image
    orig_h, orig_w, _ = img.shape
    img_resized = cv2.resize(img, resize_shape)

    # Scale GT point coordinates
    w_ratio = resize_shape[0] / orig_w
    h_ratio = resize_shape[1] / orig_h
    points_rescaled = points * [w_ratio, h_ratio]

    # Normalize image
    img_normalized = img_resized / 255.0

    # Convert to CHW tensor-friendly format
    img_tensor = np.transpose(img_normalized, (2, 0, 1))

    # Generate density map
    density_map = generate_density_map(img_resized, points_rescaled, sigma=sigma)
    density_map = density_map.astype(np.float32)

    # Save as .npy (much faster than reprocessing)
    np.save(os.path.join(save_img_dir, img_name.replace(".jpg", ".npy")), img_tensor)
    np.save(os.path.join(save_den_dir, img_name.replace(".jpg", ".npy")), density_map)

print("\n✔ Preprocessing Complete!")
print(f"Processed images saved in: {save_img_dir}")
print(f"Density maps saved in: {save_den_dir}")



#1 image heatmap display


#loading image and graund truth paths
img_path = r"archive (2)\ShanghaiTech\part_B\train_data\images\IMG_5.jpg"
gt_path = r"archive (2)\ShanghaiTech\part_B\train_data\ground-truth\GT_IMG_5.mat"

# Load image in RGB (converting from bgr to rgb)
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Load .mat ground truth
gt = loadmat(gt_path)
points = gt["image_info"][0][0][0][0][0]   

print("Original no. of points:", len(points))

# 2. Resize image and scale GT points

resize_shape = (256, 256)   # (width, height)=256,256

orig_h, orig_w, _ = img.shape
img_resized = cv2.resize(img, resize_shape)

w_ratio = resize_shape[0] / orig_w
h_ratio = resize_shape[1] / orig_h

# scale GT point coordinates
points_rescaled = points * [w_ratio, h_ratio]

# 3. Normalize image
img_normalized = img_resized / 255.0

# Convert to tensor 
img_tensor = torch.tensor(img_normalized, dtype=torch.float32).permute(2, 0, 1)
print("Image tensor shape:", img_tensor.shape)

# 4. Density-map generation function
def generate_density_map(image, points, sigma=4):
    """
    Generates density map where each head annotation gets a Gaussian.
    Lower sigma helps highlight individual heads more clearly.
    """
    h, w, _ = image.shape
    density = np.zeros((h, w), dtype=np.float32)

    if len(points) == 0:
        return density

    # place a "1" at each GT head location
    for x, y in points:
        x = int(min(w - 1, max(0, x)))
        y = int(min(h - 1, max(0, y)))
        density[y, x] = 1

    # blur the 1-pixel dots into smooth Gaussians
    density = gaussian_filter(density, sigma=sigma)

    return density

# 5. Generate density map
density_map = generate_density_map(img_resized, points_rescaled, sigma=4)
density_tensor = torch.tensor(density_map, dtype=torch.float32).unsqueeze(0)

print("Density map shape:", density_tensor.shape)
print("Estimated Count (sum of density map):", float(np.sum(density_map)))

# 6. Plot everything
plt.figure(figsize=(6,6))
plt.imshow(img_resized)
plt.title("Resized Image")
plt.axis("off")
plt.show()

plt.figure(figsize=(6,6))
plt.imshow(density_map, cmap='jet')
plt.colorbar(label="Density")
plt.title("Density Map (sigma=4)")
plt.axis("off")
plt.show()
