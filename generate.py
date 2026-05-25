import torch
from PIL import Image
import torchvision.transforms as transforms
from models.model import ImageCaptioningModel
from utils.config import *

def generate_caption(model, image, vocab, max_length=20):
    result = ["<SOS>"]

    for _ in range(max_length):
        indices = [vocab.stoi.get(word, 3) for word in result]
        tensor = torch.tensor(indices).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model.decoder(model.encoder(image), tensor)

        predicted = output.argmax(2)[:, -1].item()
        word = vocab.itos[predicted]

        result.append(word)
        if word == "<EOS>":
            break

    return " ".join(result)
