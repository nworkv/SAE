# Import
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from dsdl import ImageDataset
from sparseautoencoder import SparseAutoEncoder, save_model, create_chekpoint
from transformers import AutoImageProcessor, AutoModelForImageClassification
from tqdm.notebook import tqdm
import os

# Configuration
from conf import *

#####################################################################

# Load VIT directly
processor = AutoImageProcessor.from_pretrained("WinKawaks/vit-small-patch16-224")
model = AutoModelForImageClassification.from_pretrained("WinKawaks/vit-small-patch16-224")
model.config.output_hidden_states = True
model.eval()
for param in model.parameters():
  param.requires_grad = False
model = model.to(DEVICE)

# Creating dataset and dataloader
exts = (".jpg", ".jpeg", ".png", ".bmp", ".gif")
images = [ os.path.join(IMAGE_FOLDER, img) for img in os.listdir(IMAGE_FOLDER)
                                                                      if img.lower().endswith(exts)]
dataset = ImageDataset(images, processor)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

# Instantiate the model
dim_model = model.config.hidden_size
hidden_dim = HIDDEN_MULTIPLIER * dim_model
sae = SparseAutoEncoder(dim_model, hidden_dim, SPARSITY_K).to(DEVICE)
optimizer = torch.optim.Adam(sae.parameters(), lr=LR)
scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=SHEDULER_LAMBDA_PARAM)

######################################################################

# Training
losses = []
for epoch in range(1, NUM_EPOCHS + 1):
    epoch_loss = 0.0
    for batch in tqdm(dataloader, desc=f"Epoch {epoch}/{NUM_EPOCHS}"):
        batch = batch.to(DEVICE)
        with torch.no_grad():
            out = model(batch)
            hs = out.hidden_states
            feats = hs[LAYER_INDEX][:, 0, :]  # CLS token [B, d_model]

        # Forward SAE
        optimizer.zero_grad()
        recon = sae(feats)[0]
        loss = F.mse_loss(recon, feats)

        loss.backward()
        optimizer.step()
        scheduler.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(dataloader)
    print("Loss: ", avg_loss)
    losses.append(avg_loss)

    #saving
    save_model(sae, OUTPUT_DIR, "sae")
    if epoch % 5 == 0:
      create_chekpoint(sae, optimizer, loss, scheduler, epoch, OUTPUT_DIR)