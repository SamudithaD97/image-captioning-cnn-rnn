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
```

---

# Features

- CNN + RNN image captioning pipeline
- Attention-based caption generation
- Pretrained ResNet-50 encoder
- BLEU score evaluation
- Dynamic dataset filtering
- Portable training and inference scripts

---

# Requirements

Install required packages using:

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install torch torchvision pandas pillow nltk tqdm
```

Download NLTK tokenizer once:

```python
import nltk
nltk.download('punkt')
```

---

# Dataset Setup

Download the Flickr8k dataset and organize files as follows:

```text
data/
├── Flickr8k_Dataset/
└── Flickr8k_text/
```

Required folder names:

- `Flickr8k_Dataset`
- `Flickr8k_text`

---

# Training the Model

Run the training script:

```bash
python train.py
```

Training process includes:

- Dataset loading
- CNN + RNN training
- Loss calculation
- BLEU score evaluation
- Model checkpoint saving

Trained model is saved as:

```text
model.pth
```

---

# Generating Captions

Generate captions for images using:

```bash
python generate.py
```

---

# Model Architecture

| Component | Model |
|---|---|
| Encoder | ResNet-50 (Pretrained) |
| Decoder | LSTM |
| Attention | Bahdanau-style Attention |
| Loss Function | CrossEntropyLoss |
| Optimizer | Adam |

---

# Results

- Training loss reduced from approximately `8.0` to `2.1`
- Model generates semantically meaningful captions
- Attention mechanism improves caption relevance
- BLEU scores improve with smoothing techniques

---

# Known Limitations

- Some captions may be generic
- Fine-grained object recognition is limited
- BLEU score may not fully represent semantic quality

---

# Notes

- Missing images are automatically filtered
- Code supports both local and server execution
- File paths are dynamically handled for portability

---

# Technologies Used

- Python
- PyTorch
- Torchvision
- NLTK
- Pandas
- Pillow

---

# License

This project was developed for academic purposes.
