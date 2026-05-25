import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from utils.vocabulary import Vocabulary
import torchvision.transforms as transforms


class FlickrDataset(Dataset):
    def __init__(self, root_dir, captions_file, freq_threshold=5):
        self.root_dir = root_dir

        # Load captions file
        self.df = pd.read_csv(
            captions_file,
            sep='\t',
            names=["image", "caption"]
        )

        # Remove #0, #1, etc from image names
        self.df["image"] = self.df["image"].apply(lambda x: x.split("#")[0])

        # 🔥 Keep only images that actually exist
        self.df = self.df[self.df["image"].apply(
            lambda x: os.path.exists(os.path.join(self.root_dir, x))
        )]

        self.df = self.df.reset_index(drop=True)

        print(f"✅ Total valid samples: {len(self.df)}")

        # Build vocabulary
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocab(self.df["caption"].tolist())

        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        caption = self.df.iloc[index]["caption"]
        img_id = self.df.iloc[index]["image"]

        img_path = os.path.join(self.root_dir, img_id)

        # Load image safely
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        # Convert caption to numbers
        numericalized = [self.vocab.stoi["<SOS>"]]

        numericalized += self.vocab.numericalize(caption)

        numericalized.append(self.vocab.stoi["<EOS>"])

        return img, torch.tensor(numericalized)