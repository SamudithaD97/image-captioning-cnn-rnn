# Image Captioning using CNN + RNN with Attention

This project implements an **Image Captioning System** that automatically generates textual descriptions for images using deep learning techniques.

The architecture combines:

- A **CNN encoder** for image feature extraction
- An **LSTM decoder** for caption generation
- An **Attention mechanism** to improve caption quality and relevance

The model is trained on the **Flickr8k dataset** using **PyTorch**.

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

- End-to-end CNN + RNN image captioning pipeline
- Attention-based caption generation
- Pretrained ResNet-50 encoder
- BLEU score evaluation
- Dynamic dataset filtering
- Portable training and inference scripts

---

# Requirements

Install the required packages using:

```bash
pip install -r requirements.txt
```

Or install them manually:

```bash
pip install torch torchvision pandas pillow nltk tqdm
```

Download the NLTK tokenizer once before training:

```python
import nltk
nltk.download('punkt')
```

---

# Dataset Setup

Download the **Flickr8k dataset** and organize the files as follows:

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

The training process includes:

- Dataset loading and preprocessing
- CNN + RNN model training
- Loss calculation
- BLEU score evaluation
- Model checkpoint saving

The trained model will be saved as:

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
| Attention | Bahdanau Attention |
| Loss Function | CrossEntropyLoss |
| Optimizer | Adam |

---

# Results

- Training loss decreased from approximately **8.0** to **2.1**
- The model generates semantically meaningful captions
- The attention mechanism improves caption relevance
- BLEU scores improve with smoothing techniques

---

# Known Limitations

- Some generated captions may still be generic
- Fine-grained object recognition remains limited
- BLEU score may not fully capture semantic quality

---

# Notes

- Missing images are automatically filtered during preprocessing
- The code supports both local and server-based execution
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

