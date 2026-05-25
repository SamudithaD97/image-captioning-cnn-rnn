import os
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🔥 Get project root directory automatically
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Build paths relative to project root
IMAGE_DIR = os.path.join(BASE_DIR, "data", "Flickr8k_Dataset")
CAPTION_FILE = os.path.join(BASE_DIR, "data", "Flickr8k_text", "Flickr8k.token.txt")

EMBED_SIZE = 256
HIDDEN_SIZE = 256
NUM_EPOCHS = 10
LR = 3e-4
BATCH_SIZE = 32