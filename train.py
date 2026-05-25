from utils.config import IMAGE_DIR, CAPTION_FILE
import os

print("IMAGE_DIR:", IMAGE_DIR)
print("CAPTION_FILE:", CAPTION_FILE)
print("IMAGE_DIR exists?", os.path.exists(IMAGE_DIR))
print("CAPTION_FILE exists?", os.path.exists(CAPTION_FILE))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.dataset import FlickrDataset
from utils.config import *
from models.model import ImageCaptioningModel
from utils.bleu import calculate_bleu

from torch.nn.utils.rnn import pad_sequence


# ---------------------------
# Collate function (IMPORTANT)
# ---------------------------
def collate_fn(batch):
    imgs = [item[0] for item in batch]
    captions = [item[1] for item in batch]

    imgs = torch.stack(imgs)

    captions = pad_sequence(captions, batch_first=True, padding_value=0)

    return imgs, captions


# ---------------------------
# Load dataset
# ---------------------------
dataset = FlickrDataset(IMAGE_DIR, CAPTION_FILE)
loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)

vocab = dataset.vocab
vocab_size = len(vocab.itos)

print(f"Vocabulary size: {vocab_size}")


# ---------------------------
# Initialize model
# ---------------------------
model = ImageCaptioningModel(
    EMBED_SIZE,
    HIDDEN_SIZE,
    vocab_size
).to(DEVICE)

criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=LR)


# ---------------------------
# Training loop
# ---------------------------
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0

    for idx, (imgs, captions) in enumerate(loader):
        imgs = imgs.to(DEVICE)
        captions = captions.to(DEVICE)

        # Input = captions without last token
        inputs = captions[:, :-1]

        # Target = captions without first token
        targets = captions[:, 1:]

        outputs = model(imgs, inputs)

        loss = criterion(
            outputs.reshape(-1, vocab_size),
            targets.reshape(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if idx % 100 == 0:
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Step [{idx}] Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(loader)
    print(f"Epoch [{epoch+1}] Average Loss: {avg_loss:.4f}")


    # ---------------------------
    # BLEU evaluation (simple)
    # ---------------------------
    model.eval()

    sample_img, sample_caption = next(iter(loader))
    sample_img = sample_img.to(DEVICE)

    # simple greedy generation
    def generate_caption(model, image, vocab, max_len=20):
        result = ["<SOS>"]

        for _ in range(max_len):
            indices = [vocab.stoi.get(word, 3) for word in result]
            tensor = torch.tensor(indices).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                output = model(image, tensor)

            predicted = output.argmax(2)[:, -1].item()
            word = vocab.itos[predicted]

            result.append(word)

            if word == "<EOS>":
                break

        return " ".join(result[1:-1])

    predicted_caption = generate_caption(model, sample_img[0].unsqueeze(0), vocab)

    reference_caption = " ".join([
        vocab.itos[idx.item()]
        for idx in sample_caption[0]
        if idx.item() not in [0]
    ])

    bleu_score = calculate_bleu(reference_caption, predicted_caption)

    print(f"Sample Prediction: {predicted_caption}")
    print(f"Reference: {reference_caption}")
    print(f"BLEU Score: {bleu_score:.4f}")


# ---------------------------
# Save model
# ---------------------------
torch.save(model.state_dict(), "model.pth")
print("Model saved as model.pth")