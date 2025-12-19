import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torchvision import models
from tqdm import tqdm


# CSRNET MODEL (SAME AS BEFORE)
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

        self.output_layer = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x

# DATASET
MAX_SAMPLES = 100

class ShanghaiPreproDataset(Dataset):
    def __init__(self, prepro_dir):
        self.files = sorted([
            os.path.join(prepro_dir, f)
            for f in os.listdir(prepro_dir)
            if f.endswith(".pt")
        ])[:MAX_SAMPLES]

        print(f"Using {len(self.files)} samples from {prepro_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx], map_location="cpu")
        return data["image"].float(), data["gt"].float()


# FINETUNING FUNCTION
def finetune():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # PART-A DATA
    prepro_dir = r"archive (2)/ShanghaiTech/preprocessed/part_A/train_data"

    dataset = ShanghaiPreproDataset(prepro_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    model = CSRNet().to(device)
       
       
    # LOAD YOUR ALREADY TRAINED PART-B MODEL
    model.load_state_dict(torch.load("model_4.pth", map_location=device))
    print("Loaded weights from model_4.pth")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-6)  # smaller LR for finetune

    EPOCHS = 3   # small number is enough

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

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
        print(f"Epoch {epoch+1}: Batch Loss 1 = {avg_loss:.6f}")

    # SAVE NEW MODEL
    torch.save(model.state_dict(), "model_5.pth")
    print("\nFine-tuned model saved as model_5.pth")


# RUN
if __name__ == "__main__":
    finetune()