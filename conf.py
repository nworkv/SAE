# -*- coding: utf-8 -*-
import torch

IMAGE_FOLDER ='/kaggle/input/flickr2k/Flickr2K'
OUTPUT_DIR = "/content/gdrive/MyDrive/SparceAutoEncoder"
LOAD_PATH = "/content/gdrive/MyDrive/SparceAutoEncoder/sae_20250712_104941"  
BATCH_SIZE = 30
NUM_EPOCHS = 10
LR = 0.002
SHEDULER_LAMBDA_PARAM = 0.7
LAYER_INDEX = 6
HIDDEN_MULTIPLIER = 16         # Latent dimension = 16 × input_dim
SPARSITY_K = 64                # Top-K sparsity
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")