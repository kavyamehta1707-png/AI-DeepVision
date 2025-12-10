import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torchvision import models
from tqdm import tqdm

#CSRNET MODEL
class CSRNet(nn.Module):
    def __init__(self):
        super(CSRNet, self).__init__()

        vgg = models.vgg16(pretrained=True)
        self.frontend = nn.Sequential(*list(vgg.features.children())[:23])

        self.backend = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True)
        )

        self.output_layer = nn.Sequential(
            nn.Conv2d(64, 1, 1)
        )

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x  
  
#  DATASET (ONLY PARTIAL DATA TO REDUCE TRAIN TIME)
MAX_SAMPLES = 100   # change to reduce / increase dataset size

class ShanghaiPreproDataset(Dataset):
    def __init__(self, prepro_dir):
        self.prepro_dir = prepro_dir  

        all_files = [f for f in os.listdir(prepro_dir) if f.endswith(".pt")]

        # Subset the dataset for faster training
        self.files = all_files[:MAX_SAMPLES]  
        print(f"Using {len(self.files)} preprocessed samples out of {len(all_files)}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.prepro_dir, self.files[idx])
        data = torch.load(path)

        img = data["image"].float()
        gt  = data["gt"].float()

        return img, gt

#  TRAINING LOOP
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using:", device)

    prepro_dir = r"archive (2)/ShanghaiTech/part_B/train_data/images/images_prepro"

    dataset = ShanghaiPreproDataset(prepro_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    model = CSRNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    EPOCHS = 5   # <<<<<< you only want 5 epochs

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for img, gt in tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            img = img.to(device)
            gt = gt.to(device)

            optimizer.zero_grad()
            pred = model(img)

            loss = criterion(pred, gt)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}: Batch Loss 1 = {avg_loss:.4f}")

    # Save trained model
    torch.save(model.state_dict(), "model_3.pth")
    print("Training completed. Model saved as model_3.pth")
      
if __name__ == "__main__":
    train()