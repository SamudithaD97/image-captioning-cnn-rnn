# Image Captioning using CNN + RNN with Attention

This project implements an image captioning system that generates textual descriptions for images using deep learning.

The architecture combines:

- CNN encoder for image feature extraction
- LSTM decoder for caption generation
- Attention mechanism for improved caption quality

The model is trained on the Flickr8k dataset using PyTorch.

---

# Project Structure

```text
project/
│
├── data/
│   ├── Flickr8k_Dataset/
│   └── Flickr8k_text/
│
├── models/
│   ├── encoder.py
│   ├── decoder.py
│   ├── attention.py
│   └── model.py
│
├── utils/
│   ├── dataset.py
│   ├── vocabulary.py
│   ├── config.py
│   └── bleu.py
│
├── train.py
├── generate.py
├── requirements.txt
└── README.md
